use rand::{seq::SliceRandom,thread_rng};
use rand_distr::{Exp,Distribution};
use std::fs::File;
use std::io::prelude::*;
use ndarray::{prelude::*,Array2};
use std::io::{BufRead, BufReader};


const NUM_RUNS: usize = 1000;
const NUM_EPOCHS: usize = 2000;

#[derive(Debug)]
struct Device {
    x: Array2<f32>,
    y: Array2<f32>,
    gradient: Array2<f32>,
    comm_rate_up: f64,
    comm_rate_down: f64,
    mac_rate: f64,
    epoch_time: f64,
    epoch_time_uncoded: f64
}

impl Clone for Device {
    fn clone(&self) -> Device {
        let mut x = Array2::zeros((self.x.shape()[0],self.x.shape()[1]));
        x.assign(&self.x);
        let mut y = Array2::zeros((self.y.shape()[0],self.y.shape()[1]));
        y.assign(&self.y);
        let mut gradient = Array2::zeros((self.gradient.shape()[0],self.gradient.shape()[1]));
        gradient.assign(&self.gradient);
        
        Device{x, y, gradient,
            comm_rate_up: self.comm_rate_up,
            comm_rate_down: self.comm_rate_down,
            mac_rate: self.mac_rate,
            epoch_time: self.epoch_time,
            epoch_time_uncoded: self.epoch_time_uncoded
        }
    }
}

impl Device {
    fn new(f: usize, n: usize) -> Device {
        Device{
            x: Array2::zeros((n,f)),
            y: Array2::zeros((n,10)),
            gradient: Array2::zeros((f,10)),
            comm_rate_up: 0.0,
            comm_rate_down: 0.0,
            mac_rate: 0.0,
            epoch_time: 0.0,
            epoch_time_uncoded: 0.0
        }
    }
}


fn main() {

    //rayon::ThreadPoolBuilder::new().num_threads(12).build_global().unwrap();

    // System parameters
    let d = 120;    
    let f = 2_000.0;
    let n = 60_000/d;
    let master_macr = 8_240_000_000_000.0;
    let bandwidth_up = 5_000_000.0;
    let bandwidth_down = 10_000_000.0;
    let pi: f64 = 0.1;
    let num_dropouts = 0;
    let mut compute_accuracy = false;
    let mac_rates = vec![25_000_000.0, 5_000_000.0, 2_500_000.0, 1_250_000.0];

    let mut mu: f32 = 6.0;
    let lambda = 0.000_009;

    let header_overhead = 1.1;
    let num_batches = 5;
 
    let eta = 2.0;
    let mut accuracies = Vec::<f64>::new();
    let mut beta_r: Array2<f32> = Array2::zeros((f as usize,10));
    let mut train_images: Vec<Vec<f32>> = vec![vec![0.0;f as usize];60_000];
    let mut train_labels: Vec<Vec<f32>>= vec![vec![0.0;10];60_000];
    let mut transformed_test_set_x = Array2::zeros((10_000,f as usize));
    let mut transformed_test_set_y: Vec<Vec<f32>> = vec![vec![0.0;10];10_000];
    let communication_failure_prob = pi;
    let mean_communication_tries = 1.0/(1.0 - communication_failure_prob);
    let u = (d - num_dropouts) as f64;
    //let privacy_levels = vec![1,50,100,200,300,600];
    let privacy_levels = vec![1,5,10,20,30,60];

    let rng = &mut rand::thread_rng();

    // Number of threads in parallelization
    rayon::ThreadPoolBuilder::new().num_threads(30).build_global().unwrap();


    if compute_accuracy {
        /*
        let mut fi = BufReader::new(File::open("data/SortedMNISTimages.txt").unwrap());
        let sorted_images: Vec<Vec<f32>> = fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                .map(|number| number.parse().unwrap())
                .collect())
            .collect();

        let mut fi = BufReader::new(File::open("data/SortedMNISTlabel.txt").unwrap());
        let sorted_labels: Vec<Vec<f32>> = fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                    .map(|number| number.parse().unwrap())
                    .collect())
            .collect();
        */

        let fi = BufReader::new(File::open("data/MNISTvINTEL_trainImages.txt").unwrap());
        train_images = fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                .map(|number| number.parse().unwrap())
                .collect())
            .collect();

        let fi = BufReader::new(File::open("data/MNISTvINTEL_trainLabels.txt").unwrap());
        train_labels = fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                    .map(|number| number.parse().unwrap())
                    .collect())
            .collect();

        let fi = BufReader::new(File::open("data/MNISTvINTEL_testImages.txt").unwrap());
        let test_set_x: Vec<Vec<f32>> = fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                .map(|number| number.parse().unwrap())
                .collect())
            .collect();
        for i in 0..10_000 {
            for j in 0..f as usize{
                transformed_test_set_x[[i,j]] = test_set_x[i][j];
            }
        }

        let fi = BufReader::new(File::open("data/MNISTvINTEL_testLabels.txt").unwrap());
        transformed_test_set_y= fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                    .map(|number| number.parse().unwrap())
                    .collect())
            .collect();
    }

    


    // Simulate multiple training runs
    let mut run = 0;
    let mut average_times = vec![vec![0.0;NUM_EPOCHS + 1];privacy_levels.len()];
    let mut average_times_uncoded = vec![0.0;NUM_EPOCHS+1];
    while run < NUM_RUNS{
        

        // Initialize the devices
        let mut devices = vec![Device::new(f as usize,n); d];

        if compute_accuracy {
            // Assign data to devices

            let mut x_data_iter = train_images.iter();
            let mut y_data_iter = train_labels.iter();
            
            for dev in devices.iter_mut() {
                for i in 0..n{
                    let x_vec = x_data_iter.next().unwrap();
                    let y_vec = y_data_iter.next().unwrap();
                    for j in 0..f as usize{
                        dev.x[[i,j]] = x_vec[j];
                    }
                    for j in 0..10{
                        dev.y[[i,j]] = y_vec[j];
                    }
                }
            }
            accuracies.push(0.1);
        }

        
        for dev in devices.iter_mut() {
            dev.mac_rate = *mac_rates.choose(rng).unwrap();

            dev.comm_rate_up = bandwidth_up;
            dev.comm_rate_down = bandwidth_down;
        }
        
        
    
        let mut time = 0.0;
        let mut time_uncoded = 0.0;

    
        let mut epoch_runs = 1;

        while epoch_runs <= NUM_EPOCHS{
            
            if epoch_runs == 200 || epoch_runs == 350 {
                mu *= 0.8;
            }

            for (i,z) in privacy_levels.iter().enumerate(){

                let t = *z as f64 + 1.0;
            
                // Computation of Gradients at the devices
                for dev in devices.iter_mut(){
                    

                    let computation_size = 2.0 * 10.0 * (n/num_batches) as f64 * f;
                    let gamma = dev.mac_rate / (eta * computation_size);
                    let rand_comp_time = Exp::new(gamma).unwrap();

                    let communication_cost_one_gradient = mean_communication_tries * 32.0 * header_overhead * 10.0 * f / bandwidth_up;

                    let encoding_size = 10.0*f/(u-t)*u*d as f64;
                    let gamma = dev.mac_rate / (eta * encoding_size);
                    let rand_enc_time = Exp::new(gamma).unwrap();

                    let communication_cost_masks = 10.0*f/(u-t) * (d-1) as f64 * 32.0 * header_overhead * (1.0/bandwidth_up + 1.0/bandwidth_down);

                    dev.epoch_time = communication_cost_masks + communication_cost_one_gradient + computation_size / dev.mac_rate + rand_comp_time.sample(rng) + encoding_size / dev.mac_rate + rand_enc_time.sample(rng);
                    dev.epoch_time_uncoded = communication_cost_one_gradient + computation_size / dev.mac_rate + rand_comp_time.sample(rng);

                    if compute_accuracy && i == 0 {
                        let mut indices = (0..n).collect::<Vec<usize>>();
                        indices.shuffle(&mut thread_rng());
                        dev.gradient = dev.x.select(Axis(0),&indices[0..n/num_batches]).t().dot(&(&dev.x.select(Axis(0),&indices[0..n/num_batches]).dot(&beta_r) - &dev.y.select(Axis(0),&indices[0..n/num_batches]))).map(|g| g/(n/num_batches) as f32);
                    }
                }

                // Sort devices by the time they took
                devices.sort_by(|a, b| a.epoch_time.partial_cmp(&b.epoch_time).unwrap());

                if compute_accuracy && i == 0{

                    // Aggregate gradients
                    let mut gradient: Array2<f32> = Array2::zeros((f as usize,10));
                    for dev in devices.iter().take(d-num_dropouts) {
                        gradient = &gradient + &dev.gradient;
                    }

                    // Update phase
                    beta_r = &beta_r - &gradient.map(|x| x*mu / d as f32) - &beta_r.map(|b| b * mu * lambda);
                }

                let epoch_time = devices[d-num_dropouts-1].epoch_time;
                
                // Sort devices by the time they took
                devices.sort_by(|a, b| a.epoch_time_uncoded.partial_cmp(&b.epoch_time_uncoded).unwrap());
                let epoch_time_uncoded = devices[d-num_dropouts-1].epoch_time_uncoded;
            

                //Transmit masks
                let mut master_time = u / (u - t) * 10.0 * f * 32.0 * header_overhead * mean_communication_tries / bandwidth_up;

                // Decoding
                let n = d as f64;
                let k = (d - num_dropouts) as f64;
                master_time += 10.0*f/(u-t) * n * (2.0 * (n-k) - 1.5 + 1.5 * n.log2().ceil()) / master_macr;

                // Time at master for adding gradients
                master_time += (f*((d-num_dropouts-1)*10) as f64)/master_macr;
                master_time += Exp::new(eta / master_time).unwrap().sample(&mut thread_rng());
                
                //let master_time = master_time + Exp::new(10.0*gamma).unwrap().sample(&mut thread_rng());
                //let master_time = 0.0;
                time += epoch_time + master_time;
                average_times[i][epoch_runs] += time;
                
                let master_time_uncoded = ((d-num_dropouts-1)as f64 *f*10.0)/master_macr;
            	let master_time_uncoded = master_time_uncoded + Exp::new(eta / master_time_uncoded).unwrap().sample(&mut thread_rng());
                
                time_uncoded += epoch_time_uncoded + master_time_uncoded;
                average_times_uncoded[epoch_runs] += time_uncoded;
            }

            if compute_accuracy {
                let mut correct_guesses = 0;
                let guess = transformed_test_set_x.dot(&beta_r);
                for i in 0..10_000{
                    let mut max = guess[[i,0]];
                    let mut max_ind = 0;
                    let mut correct_ind = 11;
                    for j in 0..10 {
                        if max < guess[[i,j]] {
                            max = guess[[i,j]];
                            max_ind = j;
                        }
                        if transformed_test_set_y[i][j] > 0.0 {
                            correct_ind = j;
                        }
                    }
                    if correct_ind == max_ind {
                        correct_guesses += 1;
                    }
                    
                }
                accuracies.push(correct_guesses as f64 / 10_000.0);
            }
            epoch_runs += 1;
        }
        run += 1;
        compute_accuracy = false;
    }

    for t_i in average_times.iter_mut() {
        for t in t_i.iter_mut() {
            *t = *t/NUM_RUNS as f64;
        }
    }
    for t in average_times_uncoded.iter_mut() {
        *t = *t/(NUM_RUNS*privacy_levels.len()) as f64;
    }
        
    let file_name = format!("MNIST_light_sec_agg_times{}.dat",d);
    let mut file = File::create(file_name).expect("Couldn't create file");
    
    for z in privacy_levels.iter(){
        write!(file, "z={}\t",*z).expect("Unable to write to file");
    }
    write!(file,"\n").expect("Unable to write to file");
    
    for epoch_number in 0..=NUM_EPOCHS {
        if epoch_number < 150 || epoch_number % 5 == 0 {
            for i in 0..privacy_levels.len() {
                write!(file, "{}\t",average_times[i][epoch_number]).expect("Unable to write to file");
            }
            write!(file, "\n").expect("Unable to write to file");
        }
    }
    
    let file_name = format!("MNIST_uncoded{}.dat",d);
    let mut file = File::create(file_name).expect("Couldn't create file");
    write!(file, "time\n").expect("Unable to write to file");
    for epoch_number in 0..=NUM_EPOCHS {
        if epoch_number < 150 || epoch_number % 5 == 0 {
        	write!(file, "{}\n",average_times_uncoded[epoch_number]).expect("Unable to write to file");
        }
    }
}
