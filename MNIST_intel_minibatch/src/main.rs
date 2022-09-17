#[macro_use(s)]
extern crate ndarray;

use ndarray::{Array2};
use rand::{prelude::*,seq::SliceRandom,thread_rng};
use rand_distr::{StandardNormal,Exp,Normal,Uniform,Distribution};
use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use mnist::MnistBuilder;
use std::io::{BufRead, BufReader};

const NUM_TIMES: u32 = 100;
const NUM_RUNS: usize = 10;

#[derive(Debug)]
struct Device {
    g: Array2<f32>,
    w: Array2<f32>,
    x: Array2<f32>,
    x_tilde: Array2<f32>,
    y: Array2<f32>,
    y_tilde: Array2<f32>,
    gradient: Array2<f32>,
    l_tilde: Vec<usize>,
    exp_return: Vec<f64>,
    comm_rate_up: f64,
    comm_rate_down: f64,
    comp_time: f64,
    setup_time: f64,
}

impl Clone for Device {
    fn clone(&self) -> Device {
        let mut g = Array2::zeros((self.g.shape()[0],self.g.shape()[1]));
        g.assign(&self.g);
        let mut w = Array2::zeros((self.w.shape()[0],self.w.shape()[1]));
        w.assign(&self.w);
        let mut x = Array2::zeros((self.x.shape()[0],self.x.shape()[1]));
        x.assign(&self.x);
        let mut x_tilde = Array2::zeros((self.x_tilde.shape()[0],self.x_tilde.shape()[1]));
        x_tilde.assign(&self.x_tilde);
        let mut y = Array2::zeros((self.y.shape()[0],self.y.shape()[1]));
        y.assign(&self.y);
        let mut y_tilde = Array2::zeros((self.y_tilde.shape()[0],self.y_tilde.shape()[1]));
        y_tilde.assign(&self.y_tilde);
        let mut gradient = Array2::zeros((self.gradient.shape()[0],self.gradient.shape()[1]));
        gradient.assign(&self.gradient);
        
        Device{g, w, x, x_tilde, y, y_tilde,gradient,
            l_tilde: self.l_tilde.to_vec(),
            exp_return: self.exp_return.to_vec(),
            comm_rate_up: self.comm_rate_up,
            comm_rate_down: self.comm_rate_down,
            comp_time: self.comp_time,
            setup_time: self.setup_time
        }
    }
}

impl Device {
    fn new(c: usize, cb: usize, f: usize, n: usize, nb: usize) -> Device {
        Device{
            g: Array2::zeros((cb,nb)),
            w: Array2::zeros((nb,nb)),
            x: Array2::zeros((n,f)),
            x_tilde: Array2::zeros((c,f)),
            y: Array2::zeros((n,10)),
            y_tilde: Array2::zeros((c,10)),
            gradient: Array2::zeros((f,10)),
            l_tilde: vec![0;NUM_TIMES as usize],
            exp_return: vec![0.0;NUM_TIMES as usize],
            comm_rate_up: 0.0,
            comm_rate_down: 0.0,
            comp_time: 0.0,
            setup_time: 0.0
        }
    }
}

fn t_star_optimization(devices: &mut Vec<Device>, n: usize, times: &Vec<f64>, f: usize, num_bits_per_float: f64, header_overhead: f64, d: usize, c: usize, pi: f64, mac_ops_per_training_point: f64) -> usize {
    //devices.par_iter_mut().for_each(|dev| {
    devices.iter_mut().for_each(|dev| {
        // Need to find lt yielding maximal expected return for each time t
        //for ((time,l_tilde),exp_return) in (times.iter().zip(dev.l_tilde.iter_mut())).zip(dev.exp_return.iter_mut()){
        let dev_comm_rate = 0.5* (dev.comm_rate_up + dev.comm_rate_down);
        let dev_comp_time = dev.comp_time;
        (times.par_iter().zip(dev.l_tilde.par_iter_mut())).zip(dev.exp_return.par_iter_mut()).for_each(|((time,l_tilde),exp_return)| {
            let mut lt = 1;
            let mut max_lt = 0;
            let mut max_return = 0.0;

            let tau = 10.0 * (f as f64)*num_bits_per_float*header_overhead/dev_comm_rate;

            let mut nu_m = 0;
            while time - tau*(nu_m as f64) > 0.0 {
                nu_m += 1;
            }
            nu_m -= 1;

            if nu_m >= 2 {
                while lt <= n {

                    
                    let gamma = 2.0 * dev_comp_time/(2.0*10.0 * (f*lt) as f64);
                   
                    
                    let mut sum: f64 = 0.0;
                    
                    let mut h = (1.0-pi)*(1.0-pi)/pi;
                    let threshold = (time - (lt * 10 * f) as f64/dev_comp_time) / tau;
                    for nu in 2..=nu_m {
                        if threshold > nu as f64 {
                            if nu >= 4 {
                                h *= 1.0/((nu-2) as f64);
                            }
                            h *= pi*((nu-1) as f64);
                            let func = (lt as f64)*(1.0 - (-1.0*gamma*tau*(threshold - nu as f64)).exp());
                            sum += h*func;
                        } else {
                            break;
                        }
                    }
                    
                    if sum > max_return {
                        max_return = sum;
                        max_lt = lt;
                    }
                    lt += 1;
                }
            }
            *l_tilde = max_lt;
            *exp_return = max_return;
        //}
        });
    });

    // Find optimal epoch time t*
    let mut overall_exp_return = vec![0.0;NUM_TIMES as usize];
    for dev in devices.iter() {
        overall_exp_return = overall_exp_return.iter().zip(dev.exp_return.iter()).map(|(&x, &y)| x + y).collect();
    }
    let mut t_star = 0;
    for (t,ret) in overall_exp_return.iter().enumerate() {
        if *ret > (d*n - c) as f64 {
            t_star = t;
            break;
        }
    }
    t_star
}

fn main() {

    // System parameters
    let d = 25;    
    let f = 2000;
    let n = 60_000/d;
    let nu_comp = 0.2;
    let device_macri = 3_072_000.0;
    let master_macr = 8_240_000_000_000.0;
    let nu_link = 0.05;
    let bandwidth_down = 10_000_000.0;
    let bandwidth_up = 5_000_000.0;
    let pi: f64 = 0.1;
    let red = 0.1;
    let num_batches = 5; 
    
    let mu_start: f32 = 6.0;
    let mu_frequency = 40;
    let mu_scaling = 0.8;
    let lambda = 0.000009;

    let num_bits_per_float: f64 = 32.0;
    let header_overhead = 1.1;



    let start_lower_time: f64 = 1.0;
    let start_upper_time: f64 = 200.0;
    let mut lower_time: f64;
    let mut upper_time: f64;
    let time_indices: Vec<u32> = (0..NUM_TIMES).collect();

    let c = (red*(d*n) as f64).round() as usize;
    let mac_ops_per_training_point = f as f64;

    let file_name = format!("MNIST_intel_new_model_red_{}.dat",red);
    let mut file = File::create(file_name).expect("Couldn't create file");

    let data = MnistBuilder::new()
        .label_format_one_hot()
        .finalize();

    // Number of threads in parallelization
    //rayon::ThreadPoolBuilder::new().num_threads(30).build_global().unwrap();

    // Initialize data collector
    let mut data_collector: Vec<Vec<(f64,f64)>> = vec![vec![];NUM_RUNS];

    let mut fi = BufReader::new(File::open("data/MNISTvINTEL_trainImages.txt").unwrap());
    let train_images: Vec<Vec<f32>> = fi.lines()
        .map(|l| l.unwrap().split(char::is_whitespace)
            .map(|number| number.parse().unwrap())
            .collect())
        .collect();

    let mut fi = BufReader::new(File::open("data/MNISTvINTEL_trainLabels.txt").unwrap());
    let train_labels: Vec<Vec<f32>> = fi.lines()
        .map(|l| l.unwrap().split(char::is_whitespace)
                .map(|number| number.parse().unwrap())
                .collect())
        .collect();

    let mut fi = BufReader::new(File::open("data/MNISTvINTEL_testImages.txt").unwrap());
    let test_set_x: Vec<Vec<f32>> = fi.lines()
        .map(|l| l.unwrap().split(char::is_whitespace)
            .map(|number| number.parse().unwrap())
            .collect())
        .collect();
    let mut transformed_test_set_x = Array2::zeros((10_000,f));
    for i in 0..10_000 {
        for j in 0..f {
            transformed_test_set_x[[i,j]] = test_set_x[i][j];
        }
    }

    let mut fi = BufReader::new(File::open("data/MNISTvINTEL_testLabels.txt").unwrap());
    let transformed_test_set_y: Vec<Vec<f32>> = fi.lines()
        .map(|l| l.unwrap().split(char::is_whitespace)
                .map(|number| number.parse().unwrap())
                .collect())
        .collect();

    // Simulate multiple training runs
    let mut run = 0;
    while run < NUM_RUNS{
        
        // Reinitialize starting values
        lower_time = start_lower_time;
        upper_time = start_upper_time;
        let mut mu = mu_start;

        // Generate vector of considered epoch times in optimization
        let mut times: Vec<f64> = time_indices.iter().map(|&t| lower_time + (t as f64)*(upper_time - lower_time)/(NUM_TIMES as f64)).collect();

        

        // Generate random assignment of heterogeneities
        let mut comp_time_indices = (0..d).collect::<Vec<usize>>();
        let mut comm_time_indices = (0..d).collect::<Vec<usize>>();
        comp_time_indices.shuffle(&mut thread_rng());
        comm_time_indices.shuffle(&mut thread_rng());

        let mut mac_rates= Vec::new();
        
        for i in 0..10 {
            mac_rates.push(25_000_000.0);
        }
        for i in 0..5 {
            mac_rates.push(5_000_000.0);
        }
        for i in 0..5 {
            mac_rates.push(2_500_000.0);
        }
        for i in 0..5 {
            mac_rates.push(1_250_000.0);
        }
        
        mac_rates.shuffle(&mut thread_rng());


        /*
        let mut w_mat = vec![vec![0.0;28*28];f];
        let mut delta_vec = vec![0.0;f];
        let normal = Normal::new(0.0, 0.141421).unwrap();
        let uniform = Uniform::new(0.0,2.0*std::f64::consts::PI);
        for (w_i,delta) in w_mat.iter_mut().zip(delta_vec.iter_mut()){
            for w_ij in w_i.iter_mut(){
                *w_ij = normal.sample(&mut rand::thread_rng());
            }
            *delta = uniform.sample(&mut rand::thread_rng()) as f32;
        }
        let factor: f32 = (2.0 / f as f32).sqrt();

        // Transform test set
        let mut x_data_iter = data.tst_img.iter();
        let mut y_data_iter = data.tst_lbl.iter();
        let mut transformed_test_set_x = Array2::zeros((10_000,f));
        let mut transformed_test_set_y = vec![vec![0.0;10];10_000];

        for i in 0..10_000{
            let mut x_vec = vec![];
            for j in 0..28*28 {
                x_vec.push(*x_data_iter.next().unwrap() as f32 / 255.0) 
            }
            for j in 0..f{
                let mut inner_prod:f32 = 0.0;
                for k in 0..28*28{
                    inner_prod +=  x_vec[k] * w_mat[j][k];
                }
                transformed_test_set_x[[i,j]] = factor * (inner_prod + delta_vec[j]).cos();
            }
            for j in 0..10{
                transformed_test_set_y[i][j] = *y_data_iter.next().unwrap() as f32;
            }
        }

        let mut x_data_iter = data.trn_img.iter();
        let mut y_data_iter = data.trn_lbl.iter();
        */
        

        // Initialize the devices
        let mut devices = vec![Device::new(c,c/num_batches,f,n,n/num_batches); d];
        /*
        for (id,dev) in devices.iter_mut().enumerate() {
            for i in 0..d{
                let mut x_vec = vec![];
                for j in 0..28*28 {
                    x_vec.push(*x_data_iter.next().unwrap() as f32 / 255.0); 
                }
                for j in 0..f{
                    let mut inner_prod: f32 = 0.0;
                    for k in 0..28*28{
                        inner_prod +=  x_vec[k] * w_mat[j][k];
                    }
                    dev.x[[i,j]] = factor * (inner_prod + delta_vec[j]).cos();
                }
                for j in 0..10{
                    dev.y[[i,j]] = 1.0 * *y_data_iter.next().unwrap() as f32;
                }
            }
        }
        */
        let mut x_data_iter = train_images.iter();
        let mut y_data_iter = train_labels.iter();
        
        for (id,dev) in devices.iter_mut().enumerate() {
            for i in 0..n{
                let x_vec = x_data_iter.next().unwrap();
                let y_vec = y_data_iter.next().unwrap();
                println!("{:?}",y_vec);
                for j in 0..f{
                    dev.x[[i,j]] = x_vec[j];
                }
                for j in 0..10{
                    dev.y[[i,j]] = y_vec[j];
                }
            }
        }


        devices.par_iter_mut().enumerate().for_each(|(i,dev)| {
            let normal = Normal::new(0.0, (5.0/c as f64).powf(0.5)).unwrap();
            //let normal = Normal::new(0.0, 1.0).unwrap();
            // Generate training data
            for gi in dev.g.iter_mut() {
                //*gi = (1./c as f32).powf(0.5) * normal.sample(&mut rand::thread_rng()) as f32;
                *gi = normal.sample(&mut rand::thread_rng()) as f32;
            }
            
            // Assign heterogeneities
            //dev.comp_time = 1.0 - nu_comp;
            //dev.comp_time = device_macri*dev.comp_time.powf(comp_time_indices[i] as f64);

            dev.comp_time = mac_rates[i];

            //dev.comm_rate = 1.0 - nu_link;
            //dev.comm_rate = bandwidth*dev.comm_rate.powf(comm_time_indices[i] as f64);

            dev.comm_rate_up = bandwidth_up;
            dev.comm_rate_down = bandwidth_down;
        });

        
        

        // Optimization to obtain epoch time
        let mut t_star = t_star_optimization(&mut devices, n/num_batches, &times, f, num_bits_per_float, header_overhead, d, c/num_batches, pi, mac_ops_per_training_point);
        println!("t* = {}", times[t_star]);
        // Reiterate optimization
        let mut count = 0;
        while count < 1 {
            // Generate vector of considered epoch times in optimization
            let step = (upper_time - lower_time)/(NUM_TIMES as f64);
            lower_time = times[t_star] - step;
            upper_time = times[t_star] + step;
            times = time_indices.iter().map(|&t| lower_time + (t as f64)*(upper_time - lower_time)/(NUM_TIMES as f64)).collect();
            t_star = t_star_optimization(&mut devices, n/num_batches, &times, f, num_bits_per_float, header_overhead, d, c/num_batches, pi,mac_ops_per_training_point);
            println!("t* = {}", times[t_star]);
            count += 1;
        }


        // Encode Data
        devices.par_iter_mut().for_each(|dev| {

            let mut wi: f32 = (1.0 - dev.exp_return[t_star]/dev.l_tilde[t_star] as f64) as f32;
            if wi < 0.0 {
                wi = 0.0;
            } else {
                wi = wi.powf(0.5);
            }

            for i in 0..n/num_batches {
                if i < dev.l_tilde[t_star] {
                    dev.w[[i,i]] = wi;
                } else {
                    dev.w[[i,i]] = 1.0;
                }
            }
            let gw = dev.g.dot(&dev.w);     // gw = G*W
            for i in 0..num_batches{
                let normal = Normal::new(0.0, (5.0/c as f64).powf(0.5)).unwrap();
                //let normal = Normal::new(0.0, 1.0).unwrap();
                // Generate training data
                for gi in dev.g.iter_mut() {
                    //*gi = (1./c as f32).powf(0.5) * normal.sample(&mut rand::thread_rng()) as f32;
                    *gi = normal.sample(&mut rand::thread_rng()) as f32;
                }
                let gw = dev.g.dot(&dev.w);     // gw = G*W

                let start = i*n/num_batches;
                let end = (i+1)*n/num_batches;
                let x_tilde = gw.dot(&dev.x.slice(s![start..end,..]));
                let y_tilde = gw.dot(&dev.y.slice(s![start..end,..]));
                
                let start = i*c/num_batches;
                for j in 0..c/num_batches {
                    for k in 0..f {
                        dev.x_tilde[[start + j,k]] = x_tilde[[j,k]];
                    }
                    for k in 0..10 {
                        dev.y_tilde[[start + j,k]] = y_tilde[[j,k]];
                    }
                }
            }
            

            // Encoding and transmission time
            //let a_i = n as f64/dev.comp_time;  // n MAC operations per column
            //let mu_i = 2.0/a_i;
            //let gamma = mu_i/f as f64;
            //let tc2 = Exp::new(gamma).unwrap();

            //let tc = a_i * (f + 1) as f64/dev.comp_time + tc2.sample(&mut thread_rng()); // f+1 columns with n MAC operations each
            let tcom = statrs::distribution::Geometric::new(1.0 - pi).unwrap();
            let n = tcom.sample(&mut thread_rng());

            //dev.setup_time = tc + n * (num_bits_per_float*header_overhead * (c*(f+1)) as f64 / dev.comm_rate); // c*(f+1) floats to transmit
            dev.setup_time = n * (num_bits_per_float*header_overhead * (c*(f+10)) as f64 / dev.comm_rate_up); // c*(f+1) floats to transmit
        });

        // Initialize master
        let mut master = Device::new(c,c/num_batches,f,n,n/num_batches);
        master.comp_time = master_macr;
        for dev in devices.iter() {
            master.x_tilde = &master.x_tilde + &dev.x_tilde;
            master.y_tilde = &master.y_tilde + &dev.y_tilde;
        }

        // Training period
        let mut beta_r: Array2<f32> = Array2::zeros((f,10));
        let mut gradient: Array2<f32> = Array2::zeros((f,10));
        let mut time = 0.0;
        let mut accuracy = 0.1;
        data_collector[run].push((time,accuracy));
        
        for dev in devices.iter() {
            if dev.setup_time > time {  // Find slowest device to encode and transmit
                time = dev.setup_time;
            }
        }
        


        //println!("{}    {}", nmse, time);
        data_collector[run].push((time,accuracy));
        let mut epoch_number = 0;
        //while time < 7.0e4 && (nmse_old - nmse).abs() > 1e-7{
        //while time < 5.0e4 {
        while epoch_number < 2000 {
            /*
            if epoch_number % mu_frequency == 0 && epoch_number > 0 {
                mu *= mu_scaling;
            }*/
            if epoch_number == 200 - 1 || epoch_number == 325 - 1 {
                mu *= mu_scaling;
            }

            // Computation of Gradients at the devices
            devices.par_iter_mut().for_each(|dev| {

                let l_star = dev.l_tilde[t_star];
                let gamma = 2.0*dev.comp_time / (l_star*2*10*f) as f64;
                let tc2 = Exp::new(gamma).unwrap();

                let tc = (2*10*l_star*f) as f64/dev.comp_time + tc2.sample(&mut thread_rng());
                let tcom = statrs::distribution::Geometric::new(1.0 - pi).unwrap();
                let n1 = tcom.sample(&mut thread_rng());
                let n2 = tcom.sample(&mut thread_rng());

                let t = tc + n1 * (num_bits_per_float * header_overhead * 10.0 * f as f64 / dev.comm_rate_down) + n2 * (num_bits_per_float * header_overhead * 10.0 * f as f64 / dev.comm_rate_up);

                //println!("tc = {}, t = {}", tc, t);
                dev.gradient = Array2::zeros((f,10));
                if t < times[t_star] {
                    //println!("Non straggler");
                    let start = (epoch_number % num_batches) * n/num_batches;
                    dev.gradient = dev.x.slice(s![start..start + l_star,..]).t().dot(&(&dev.x.slice(s![start..start+l_star,..]).dot(&beta_r) - &dev.y.slice(s![start..start+l_star,..])));
                } else {
                    //println!("Straggler");
                }

            });

            // Computation at the master
            //master.gradient = master.x_tilde.t().dot(&(&master.x_tilde.dot(&beta_r) - &master.y_tilde)).map(|x| x/c as f32);
            let start = (epoch_number % num_batches)*c/num_batches;
            let end = start + c/num_batches;
            master.gradient = master.x_tilde.slice(s![start..end,..]).t().dot(&(&master.x_tilde.slice(s![start..end,..]).dot(&beta_r) - &master.y_tilde.slice(s![start..end,..])));

            let gamma = 2.0*master.comp_time / (c*2*10*f) as f64;
            let tc2 = Exp::new(10.0 * gamma).unwrap();

            let tm = (2*10*c*f) as f64/master.comp_time + tc2.sample(&mut thread_rng());
            //let tm = times[t_star];

            // Aggregate gradients
            gradient.assign(&master.gradient);
            for dev in devices.iter() {
                gradient = &gradient + &dev.gradient;
            }

            // Update phase
            beta_r = &beta_r - &gradient.map(|g| g * mu * num_batches as f32/(d*n)as f32) - &beta_r.map(|b| b*mu*lambda);

            if tm < times[t_star] {
                time += times[t_star];
            } else {
                time += tm;
            }

            let mut correct_guesses = 0;
            let guess = transformed_test_set_x.dot(&beta_r);
            let mut printed = 0;
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
                /*
                if printed < 1000 {
                    println!("{:?}",(0..10).map(|j| guess[[i,j]]).collect::<Vec<f32>>());
                    println!("{:?}",transformed_test_set_y[i].iter().map(|y| *y).collect::<Vec<f32>>());
                    println!("max_ind = {}, corresct_ind = {}, correct_guesses = {}",max_ind,correct_ind,correct_guesses);
                    println!("");
                    printed += 1;
                }
                */
                
            }
            accuracy = correct_guesses as f64 / 10_000.0;
            
            data_collector[run].push((time,accuracy));
            epoch_number += 1;
            if epoch_number % 5 == 0 {
                println!("Epoch #{}:",epoch_number);
                println!("Time: {}",time);
                println!("ACC: {}",accuracy);
                println!("");
            }
        }
        
        run += 1;
    }
    
    // Average data in data_collector
    let mut most_iterations = 0;
    for training in data_collector.iter() {
        if training.len() > most_iterations {
            most_iterations = training.len();
        }
    }
    let mut average_time = vec![0.0f64;most_iterations];
    let mut average_accuracy = vec![0.0f64;most_iterations];
    let mut num_points = vec![0.0f64;most_iterations];
    for training in data_collector.iter() {
        for (i,point) in training.iter().enumerate() {
            average_accuracy[i] = average_accuracy[i] + point.0;
            average_time[i] = average_time[i] + point.1;
            num_points[i] = num_points[i] + 1.0;
        }
    }
    average_time.iter_mut().zip(num_points.iter()).for_each(|(x,y)| *x = *x / *y);
    average_accuracy.iter_mut().zip(num_points.iter()).for_each(|(x,y)| *x = *x / *y);
    

    //println!("");
    //println!("Average values:");
    write!(file, "time\t accuracy\n").expect("Unable to write to file");
    for (t,acc) in average_accuracy.iter().zip(average_time.iter()) {
        //println!("{} {}",acc,t);
        write!(file, "{} {}\n", t, acc).expect("Unable to write to file"); 
    }
    

    // Fastest setup in data_collector
    /*
    let mut fastest_setup = data_collector[0][0].1;
    let mut simulation_with_fstest_setup = 0;
    for (i,training) in data_collector.iter().enumerate() {
        if training[0].1 < fastest_setup {
            fastest_setup = training[0].1;
            simulation_with_fstest_setup = i;
        }
    }

    write!(file, "accuracy\t time\n").expect("Unable to write to file");
    for (acc,t) in data_collector[simulation_with_fstest_setup].iter() {
        //println!("{} {}",acc,t);
        write!(file, "{} {}\n", acc, t).expect("Unable to write to file"); 
    }
    */

}
