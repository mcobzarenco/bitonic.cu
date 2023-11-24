mod bindings;

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};
use rand::Rng;
use std::{cmp::Ordering, sync::Arc, time::Instant};

use crate::bindings::Struct;

// Include the compiled PTX code as string
const BITONIC_CUDA_KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/bitonic.ptx"));
const BITONIC_MODULE_NAME: &str = "bitonic";

pub struct GpuSorter {
    device: Arc<CudaDevice>,
    local_disperse: CudaFunction,
    local_binary_merge_sort: CudaFunction,
    global_flip: CudaFunction,
    global_disperse: CudaFunction,
}

const BLOCK_SIZE: u32 = 1024;

impl GpuSorter {
    pub fn new(device_ordinal: usize) -> Result<Self, DriverError> {
        // Initialise CUDA
        let now = Instant::now();
        let device = CudaDevice::new(device_ordinal)?;
        // println!("Time taken to initialise CUDA: {:.2?}", now.elapsed());

        // Compile and load kernels
        let now = Instant::now();
        let ptx = Ptx::from_src(BITONIC_CUDA_KERNEL);
        device.load_ptx(
            ptx,
            BITONIC_MODULE_NAME,
            &[
                "local_disperse",
                "local_binary_merge_sort",
                "global_flip",
                "global_disperse",
            ],
        )?;
        // println!(
        //     "Time taken to compile and load kernels: {:.2?}",
        //     now.elapsed()
        // );

        let local_disperse = device
            .get_func(BITONIC_MODULE_NAME, "local_disperse")
            .expect("local_disperse to be defined in the cuda kernel");
        let local_binary_merge_sort = device
            .get_func(BITONIC_MODULE_NAME, "local_binary_merge_sort")
            .expect("local_binary_merge_sort to be defined in the cuda kernel");
        let global_flip = device
            .get_func(BITONIC_MODULE_NAME, "global_flip")
            .expect("global_flip to be defined in the cuda kernel");
        let global_disperse = device
            .get_func(BITONIC_MODULE_NAME, "global_disperse")
            .expect("global_disperse to be defined in the cuda kernel");

        Ok(Self {
            device,
            local_disperse,
            local_binary_merge_sort,
            global_flip,
            global_disperse,
        })
    }

    fn local_binary_merge_sort(
        &self,
        block_count: u32,
        structs: &CudaSlice<Struct>,
        height: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            self.local_binary_merge_sort.clone().launch(
                LaunchConfig {
                    grid_dim: (block_count, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: BLOCK_SIZE * 4,
                },
                (structs, height),
            )
        }
    }

    fn local_disperse(
        &self,
        block_count: u32,
        structs: &CudaSlice<Struct>,
        height: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            self.local_disperse.clone().launch(
                LaunchConfig {
                    grid_dim: (block_count, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: BLOCK_SIZE * 4,
                },
                (structs, height),
            )
        }
    }

    fn global_flip(
        &self,
        block_count: u32,
        structs: &CudaSlice<Struct>,
        height: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            self.global_flip.clone().launch(
                LaunchConfig {
                    grid_dim: (block_count, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: BLOCK_SIZE * 4,
                },
                (structs, height),
            )
        }
    }

    fn global_disperse(
        &self,
        block_count: u32,
        structs: &CudaSlice<Struct>,
        height: u32,
    ) -> Result<(), DriverError> {
        unsafe {
            self.global_disperse.clone().launch(
                LaunchConfig {
                    grid_dim: (block_count, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: BLOCK_SIZE * 4,
                },
                (structs, height),
            )
        }
    }

    fn sort_structs(&self, structs: Vec<Struct>) -> Result<Vec<Struct>, DriverError> {
        let num_elements = u32::try_from(structs.len()).unwrap();

        let block_count = num_elements / (2 * BLOCK_SIZE);
        // println!("Block count = {}", block_count);

        let now = Instant::now();
        let structs_gpu = self.device.htod_copy(structs)?;

        // println!(
        //     "Time taken to copy data host -> device: {:.2?}",
        //     now.elapsed()
        // );

        let now = Instant::now();
        let mut height = BLOCK_SIZE * 2;

        self.local_binary_merge_sort(block_count, &structs_gpu, height)?;

        // Double the height, as this happens before every flip
        height *= 2;

        while height <= num_elements {
            self.global_flip(block_count, &structs_gpu, height)?;

            let mut dheight = height / 2;
            while dheight > 1 {
                if dheight <= 2 * BLOCK_SIZE {
                    self.local_disperse(block_count, &structs_gpu, dheight)?;
                    break;
                } else {
                    self.global_disperse(block_count, &structs_gpu, dheight)?;
                }

                dheight /= 2;
            }
            height *= 2;
        }
        let my_structs = self.device.sync_reclaim(structs_gpu)?;

        // println!(
        //     "Time taken to sort and copy data device -> host: {:.2?}",
        //     now.elapsed()
        // );

        Ok(my_structs)
    }
}

fn generate_some_data(num_elements: usize) -> Vec<Struct> {
    let mut structs = Vec::with_capacity(num_elements);
    let mut rng = rand::thread_rng();
    for _ in 0..num_elements {
        structs.push(Struct {
            value: rng.gen::<f32>() * 200.0 - 100.0,
        });
    }
    return structs;
}

pub fn main_full() -> Result<(), DriverError> {
    let num_elements: u32 = 1 << 28;

    let now = Instant::now();
    let structs = generate_some_data(num_elements as usize);
    println!("Generate data {num_elements} time: {:.2?}", now.elapsed());

    {
        use rayon::prelude::*;
        let mut structs = structs.clone();
        let now = Instant::now();
        structs.par_sort_unstable_by(|a, b| {
            if a.value < b.value {
                Ordering::Less
            } else {
                Ordering::Greater
            }
            // a.value
            //     .partial_cmp(&b.value)
            //     .unwrap_or(std::cmp::Ordering::Less)
        });
        println!("rayon sort: {:.2?}", now.elapsed());
    }

    let now = Instant::now();
    let sorter = GpuSorter::new(0)?;
    println!("GpuSorter::new time: {:.2?}", now.elapsed());

    let now = Instant::now();
    let my_structs = sorter.sort_structs(structs)?;
    println!("Sort time: {:.2?}", now.elapsed());

    // println!(
    //     "{:?}",
    //     my_structs
    //         .clone()
    //         .into_iter()
    //         .map(|s| s.value)
    //         .collect::<Vec<_>>()
    // );

    // let my_structs = structs;
    for (index, (previous, current)) in (&my_structs[..my_structs.len() - 1])
        .iter()
        .zip((&my_structs[1..]).iter())
        .enumerate()
    {
        assert!(
            previous.value <= current.value,
            "{}: {} > {}",
            index,
            previous.value,
            current.value
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use rayon::prelude::*;

    #[test]
    fn sorts_correctly() {
        for num_elements in (8..18).map(|pow| 1 << pow) {
            dbg!(num_elements);
            for _num_retry in 0..10 {
                // Generate some date
                let structs = generate_some_data(num_elements);

                // Make a copy and sort it with rayon
                let mut rayon_sorted = structs.clone();
                rayon_sorted.par_sort_unstable_by(|a, b| {
                    if a.value < b.value {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                });

                // Sort it with the CUDA sorter
                let gpu_sorted = GpuSorter::new(0).unwrap().sort_structs(structs).unwrap();

                // Check it's the same
                assert!(
                    rayon_sorted
                        .iter()
                        .zip(gpu_sorted.iter())
                        .all(|(lhs, rhs)| lhs == rhs),
                    "rayon: {:?}\n gpu: {:?}",
                    rayon_sorted
                        .into_iter()
                        .map(|s| s.value)
                        .collect::<Vec<_>>(),
                    gpu_sorted.into_iter().map(|s| s.value).collect::<Vec<_>>(),
                );
            }
        }
    }

    #[test]
    fn run_main() {
        main_full().unwrap();
    }
}
