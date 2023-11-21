mod bindings;

use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::time::Instant;

use crate::bindings::Struct;

// Include the compiled PTX code as string
const BITONIC_CUDA_KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/bitonic.ptx"));

fn main() -> Result<(), DriverError> {
    // setup GPU device
    let now = Instant::now();

    let gpu = CudaDevice::new(0)?;

    println!("Time taken to initialise CUDA: {:.2?}", now.elapsed());

    // compile ptx
    let now = Instant::now();

    let ptx = Ptx::from_src(BITONIC_CUDA_KERNEL);
    gpu.load_ptx(ptx, "my_module", &["bitonic_sort"])?;

    println!("Time taken to compile and load PTX: {:.2?}", now.elapsed());

    // create data
    let now = Instant::now();

    let num_elements: u32 = 8;
    let mut structs = Vec::with_capacity(num_elements as usize);
    for index in 0..num_elements {
        structs.push(Struct {
            value: (num_elements - index - 1) as f32,
        });
    }

    // copy to GPU
    let structs_gpu = gpu.htod_copy(structs)?;

    println!("Time taken to initialise data: {:.2?}", now.elapsed());

    let now = Instant::now();

    let f = gpu.get_func("my_module", "bitonic_sort").unwrap();

    unsafe {
        f.launch(
            // LaunchConfig::for_num_elems(num_elements),
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (num_elements / 2, 1, 1),
                shared_mem_bytes: 0,
            },
            (&structs_gpu, num_elements),
        )
    }?;

    println!("Time taken to call kernel: {:.2?}", now.elapsed());

    let my_structs = gpu.sync_reclaim(structs_gpu)?;
    println!(
        "{:?}",
        my_structs.into_iter().map(|s| s.value).collect::<Vec<_>>()
    );

    // assert!(my_structs.iter().all(|i| i.data == [1.0; 4]));

    Ok(())
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn run_main() {
        main().unwrap();
    }
}
