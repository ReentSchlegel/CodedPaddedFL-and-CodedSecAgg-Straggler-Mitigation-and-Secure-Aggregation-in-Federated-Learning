use rand::{prelude::*};
use rand_distr::{Exp};
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufRead, BufReader};

mod field_arithmetic;
#[allow(unused_imports)]
use crate::field_arithmetic::{*};

pub type Field = u64;
#[allow(unused)]
const NUM_FRAC_BITS: usize = 24;
#[allow(unused)]
const NUM_TOTAL_BITS: usize = 48;
#[allow(unused)]
const MODULOS: u64 = 281_474_976_710_656; // 2^48


#[derive(Debug)]
pub struct Device {
    pub id: usize,
    pub epoch_time: f64,
    pub mac_rate: f64
}

impl Device {
    pub fn new(id: usize) -> Device {
        Device {
            id: id,
            epoch_time: 0.0,
            mac_rate: 0.0
        }
    }
}





fn main() {
    let num_devices = 1000;
    let mac_rates = vec![25_000_000.0, 5_000_000.0, 2_500_000.0, 1_250_000.0];
    let central_server_comp_rate = 250_000_000.0;
    let bandwidth_cs2dev = 7_500_000.0;
    let bandwidth_up = 5_000_000.0;
    let bandwidth_down = 10_000_000.0;
    let communication_failure_prob = 0.1;
    let eta = 0.5;
    let f:f64 = 2000.0;
    let num_epochs = 2000;
    let num_runs = 1000;
    let header_overhead = 1.1;
    let field_overhead = (NUM_TOTAL_BITS + NUM_FRAC_BITS) as f64 / NUM_TOTAL_BITS as f64;
    let mean_communication_tries = 1.0/(1.0 - communication_failure_prob);
    let num_data_points = 60_000;
    let mut compute_accuracy = false;
    let mut mu = 6.0;
    let lambda = 0.000_009;
    let privacy_levels = vec![1,50,100,200,300,600];


    let mut possible_numbers_of_groups = vec![1];
    possible_numbers_of_groups.append(& mut divisors::get_divisors(num_devices));
    possible_numbers_of_groups.push(num_devices);

    // Setup RNG
    let rng = &mut rand::thread_rng();

    let mut xtx_field:Vec<Vec<Field>> = vec![vec![0;f as usize];f as usize];
    let mut x_field:Vec<Vec<Field>> = vec![vec![0;f as usize];num_data_points as usize];
    let mut y_field:Vec<Vec<Field>> = vec![vec![0;10];num_data_points as usize];
    let mut x_test_field:Vec<Vec<Field>> = vec![vec![0;f as usize];10_000];
    let mut y_test_field:Vec<Vec<Field>> = vec![vec![0;10];10_000];
    let mut grad1:Vec<Vec<Field>> = vec![vec![0;10];f as usize];
    let mut beta:Vec<Vec<Field>> = vec![vec![0;10];f as usize];
    let mut accuracies = Vec::<f64>::new();

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
        x_field = fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                .map(|number| from_num(number.parse().unwrap()))
                .collect())
            .collect();
        

        xtx_field = mat_mat_mul(&transpose(&x_field),&x_field);

        let fi = BufReader::new(File::open("data/MNISTvINTEL_trainLabels.txt").unwrap());
        y_field = fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                    .map(|number| from_num(number.parse().unwrap()))
                    .collect())
            .collect();

        let fi = BufReader::new(File::open("data/MNISTvINTEL_testImages.txt").unwrap());
        x_test_field = fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                .map(|number| from_num(number.parse().unwrap()))
                .collect())
            .collect();

        let fi = BufReader::new(File::open("data/MNISTvINTEL_testLabels.txt").unwrap());
        y_test_field = fi.lines()
            .map(|l| l.unwrap().split(char::is_whitespace)
                    .map(|number| from_num(number.parse().unwrap()))
                    .collect())
            .collect();
    }

    let mut best_times: Vec<Vec<f64>> = vec![vec![-1.0;num_epochs + 1];num_devices];
    let mut old_scheme_best_times: Vec<f64> = vec![-1.0;num_epochs + 1];
    let mut old_scheme_one_group_times: Vec<f64> = vec![-1.0;num_epochs + 1];
    let mut secure_scheme_one_group_times: Vec<Vec<f64>> = vec![vec![-1.0;num_epochs + 1];num_devices];

    for num_groups in possible_numbers_of_groups{
        let mut groups: Vec<Vec<Device>> = Vec::new();
        for _ in 0..num_groups {
            let new_group: Vec<Device> = Vec::new();
            groups.push(new_group);
        }
        for device_id in 0..num_devices {
            groups[device_id % num_groups].push(Device::new(device_id));
        }
        let num_devices_per_group = num_devices / num_groups;

        let mut time_collector: Vec<Vec<f64>> = vec![vec![0.0;num_epochs + 1];num_devices_per_group];
        let mut old_scheme_time_collector: Vec<Vec<f64>> = vec![vec![0.0;num_epochs + 1];num_devices_per_group];
        
        for _ in 0..num_runs{

            // Initialization
            for group in groups.iter_mut(){
                for device in group.iter_mut() {
                    device.epoch_time = 0.0;
                    device.mac_rate = *mac_rates.choose(rng).unwrap();
                }
            }

            if compute_accuracy {
                grad1 = mat_sca_mul(&mat_mat_mul(&transpose(&x_field),&y_field),from_num(-1.0));
            }

            // Data Sharing
            let sharing_time = (num_devices_per_group - 1) as f64 * (10.0 * f + f * (f + 1.0) / 2.0) * header_overhead * field_overhead * NUM_TOTAL_BITS as f64 * mean_communication_tries * 2.0 / bandwidth_cs2dev;
            let mut super_group_sums = vec![sharing_time;num_devices_per_group];

            for dev in 0..num_devices_per_group{
                time_collector[dev][0] += super_group_sums[dev];
            }
            
            let mut old_scheme_super_group_sums = vec![0.0;num_devices_per_group];
            for (i,time) in old_scheme_time_collector.iter_mut().enumerate(){
                let alpha = num_devices_per_group - i;
                let sharing_time = (alpha - 1) as f64 * (10.0 * f + f * (f + 1.0) / 2.0) * header_overhead * NUM_TOTAL_BITS as f64 * mean_communication_tries * 2.0 / bandwidth_cs2dev;
                time[0] = sharing_time;
                old_scheme_super_group_sums[i] = sharing_time;
            }

            if compute_accuracy {
                accuracies.push(0.1);
            }

            // Learning Phase
            for epoch_number in 1..=num_epochs {
                // Compute accuracy
                if compute_accuracy {
                    if epoch_number % 20 == 0{
                        println!("Running epoch {}/{}",epoch_number,num_epochs);
                    }
                    /*
                    if epoch_number == 200 || epoch_number == 350 {
                        mu *= 0.8;
                    }
                    */
                    if epoch_number % 80 == 0{
                        mu *= 0.9;
                    }
                    let gradient = mat_mat_add(&grad1, &mat_mat_mul(&xtx_field, &beta));
                    beta = mat_mat_sub(&mat_sca_mul(&beta,from_num(1.0-mu*lambda)), &mat_sca_mul(&gradient, from_num(mu / num_data_points as f64)));
                    let guess = mat_mat_mul(&x_test_field, &beta);
                    let mut correct_guesses = 0;
                    for i in 0..10_000{
                        let mut max = to_float(guess[i][0]);
                        let mut max_ind = 0;
                        let mut correct_ind = 11;
                        for j in 0..10 {
                            if max < to_float(guess[i][j]) {
                                max = to_float(guess[i][j]);
                                max_ind = j;
                            }
                            if to_float(y_test_field[i][j]) > 0.0 {
                                correct_ind = j;
                            }
                        }
                        if correct_ind == max_ind {
                            correct_guesses += 1;
                        }
                    }
                    accuracies.push(correct_guesses as f64 / 10_000.0);
                }

                // Computation Times
                for group in groups.iter_mut(){
                    for device in group.iter_mut() {
                        let computation_size = 10.0 * f * f * field_overhead;
                        let gamma = device.mac_rate / (eta * computation_size);
                        let rand_comp_time = Exp::new(gamma).unwrap();
                        device.epoch_time = computation_size / device.mac_rate + rand_comp_time.sample(rng);
                    }
                }

                // Aggregate Computation Times
                let mut super_group_times = vec![0.0f64;num_devices_per_group];
                for (time_i,super_group_time) in super_group_times.iter_mut().enumerate() {
                    for group in groups.iter() {
                        if group[time_i].epoch_time > *super_group_time {
                            *super_group_time = group[time_i].epoch_time;
                        }
                    }
                }
                super_group_times.sort_by(|a, b| a.partial_cmp(&b).unwrap());

                let mut old_super_group_times = vec![0.0f64;num_devices_per_group];
                for group in groups.iter(){
                    let mut group_times = vec![0.0;group.len()];
                    for (i,dev) in group.iter().enumerate() {
                        group_times[i] = dev.epoch_time;
                    }
                    group_times.sort_by(|a,b| a.partial_cmp(&b).unwrap());
                    for (max,cur) in old_super_group_times.iter_mut().zip(group_times.iter()) {
                        if *cur > *max {
                            *max = *cur;
                        }
                    }
                }

                // Gradient Sharing
                let communication_cost_one_gradient = mean_communication_tries * field_overhead * NUM_TOTAL_BITS as f64 * header_overhead * 10.0 * f * (1.0 / bandwidth_up + 1.0 / bandwidth_down);
                for time in super_group_times.iter_mut(){
                    *time += communication_cost_one_gradient * ((num_groups as f64).log2().floor() + 1.0);
                }
                let communication_cost_one_gradient = mean_communication_tries * NUM_TOTAL_BITS as f64 * header_overhead * 10.0 * f * (1.0 / bandwidth_up + 1.0 / bandwidth_down);
                for time in old_super_group_times.iter_mut(){
                    *time += communication_cost_one_gradient;
                }

                // Decoding
                for (k,time) in super_group_times.iter_mut().enumerate() {
                    let n = num_devices_per_group as f64;
                    let k = (k + 1) as f64;
                    *time += 10.0 * f * n * (2.0 * (n-k) - 1.5 + 1.5 * n.log2().ceil()) * field_overhead / central_server_comp_rate;
                }
                for (k,time) in old_super_group_times.iter_mut().enumerate() {
                    let d = num_devices_per_group;
                    let alpha = d - k;
                    *time += ((d - alpha + 1) as f64 * 10.0 * f * f) / central_server_comp_rate;
                }


                // Total Time Aggregation
                super_group_sums = super_group_sums.iter().zip(super_group_times.iter()).map(|(&a, &b)| a + b).collect();
                for dev in 0..num_devices_per_group{
                    time_collector[dev][epoch_number] += super_group_sums[dev];
                }
                old_scheme_super_group_sums = old_scheme_super_group_sums.iter().zip(old_super_group_times.iter()).map(|(&a, &b)| a + b).collect();
                for dev in 0..num_devices_per_group{
                    old_scheme_time_collector[dev][epoch_number] += super_group_sums[dev];
                }
                
            }

            // Write accuracy to file
            if compute_accuracy {
                let file_name = format!("MNIST_secure_accuracy{}_{}.dat",NUM_TOTAL_BITS,NUM_FRAC_BITS);
                let mut file = File::create(file_name).expect("Couldn't create file");
                write!(file, "Accuracy\n").expect("Unable to write to file");
                for epoch_number in 0..=num_epochs {
                    if epoch_number < 150 || epoch_number % 5 == 0 {
                        write!(file, "{}\n",accuracies[epoch_number]).expect("Unable to write to file");
                    }
                }
            }
            compute_accuracy = false;
        }
        time_collector = time_collector.iter().map(|row| row.iter().map(|a| a/num_runs as f64).collect()).collect();
        old_scheme_time_collector = old_scheme_time_collector.iter().map(|row| row.iter().map(|a| a/num_runs as f64).collect()).collect();
        for k in 0..num_devices_per_group {
            for epoch_number in 0..=num_epochs {
                for k_p in k..num_devices_per_group {
                    if time_collector[k_p][epoch_number] < best_times[k][epoch_number] || best_times[k][epoch_number] < 0.0 {
                        best_times[k][epoch_number] = time_collector[k_p][epoch_number];
                    }
                }
                if old_scheme_time_collector[k][epoch_number] < old_scheme_best_times[epoch_number] || old_scheme_best_times[epoch_number] < 0.0 {
                    old_scheme_best_times[epoch_number] = old_scheme_time_collector[k][epoch_number];
                } 
            }
        }
        if num_groups == 1 {
            for (t1,t2) in old_scheme_one_group_times.iter_mut().zip(old_scheme_best_times.iter()) {
                *t1 = *t2;
            }
            for k in 0..num_devices_per_group {
                for (t1,t2) in secure_scheme_one_group_times[k].iter_mut().zip(best_times[k].iter()) {
                    *t1 = *t2;
                }
            }
        }
    }
    println!("Finished!");
    let file_name = format!("MNIST_secure_times{}.dat",num_devices);
    let mut file = File::create(file_name).expect("Couldn't create file");
    for z in privacy_levels.iter(){
        write!(file, "z={}\t",*z).expect("Unable to write to file");
    }
    write!(file,"\n").expect("Unable to write to file");
    
    for epoch_number in 0..=num_epochs {
        if epoch_number < 150 || epoch_number % 5 == 0 {
            for z in privacy_levels.iter() {
                write!(file, "{}\t",best_times[*z + 1][epoch_number]).expect("Unable to write to file");
            }
            write!(file, "\n").expect("Unable to write to file");
        }
    }
    let file_name = format!("MNIST_old_scheme_times{}.dat",num_devices);
    let mut file = File::create(file_name).expect("Couldn't create file");
    write!(file,"times\n").expect("Unable to write to file");
    for epoch_number in 0..=num_epochs {
        if epoch_number < 150 || epoch_number % 5 == 0 {
            write!(file, "{}\n",old_scheme_best_times[epoch_number]).expect("Unable to write to file");
        }
    }
    let file_name = format!("MNIST_old_scheme_one_group_times{}.dat",num_devices);
    let mut file = File::create(file_name).expect("Couldn't create file");
    write!(file,"times\n").expect("Unable to write to file");
    for epoch_number in 0..=num_epochs {
        if epoch_number < 150 || epoch_number % 5 == 0 {
            write!(file, "{}\n",old_scheme_one_group_times[epoch_number]).expect("Unable to write to file");
        }
    }
    let file_name = format!("MNIST_secure_scheme_one_group_times{}.dat",num_devices);
    let mut file = File::create(file_name).expect("Couldn't create file");
    write!(file,"times\n").expect("Unable to write to file");
    for epoch_number in 0..=num_epochs {
        if epoch_number < 150 || epoch_number % 5 == 0 {
            for z in privacy_levels.iter() {
                write!(file, "{}\t",secure_scheme_one_group_times[*z + 1][epoch_number]).expect("Unable to write to file");
            }
            write!(file, "\n").expect("Unable to write to file");
        }
    }
}
