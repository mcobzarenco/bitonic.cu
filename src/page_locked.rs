use std::{
    alloc::{AllocError, Allocator, Layout},
    ffi::c_void,
    ptr::NonNull,
    slice,
};

#[derive(Debug, Clone)]
pub struct CudaHostAllocator;

unsafe impl Allocator for CudaHostAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        println!("  allocate({:?})", layout);
        
        let mut allocation = std::ptr::null_mut::<c_void>();
        let cu_result = unsafe {
            cudarc::driver::sys::cuMemHostAlloc(
                &mut allocation as *mut *mut c_void,
                layout.size(),
                0,
                // cudarc::driver::sys::CU_MEMHOSTALLOC_DEVICEMAP
                // cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED
            )
        };
        if cu_result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            panic!("Could not allocate page locked host mem: {:?}", cu_result);
        }
        assert_eq!(allocation.align_offset(layout.align()), 0);

        let s = unsafe { slice::from_raw_parts_mut(allocation as *mut u8, layout.size()) };

        Ok(NonNull::new(s as *mut [u8]).unwrap())
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        println!("deallocate({:?})", _layout);
        cudarc::driver::sys::cuMemFreeHost(ptr.as_ptr() as * mut c_void); 
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use cudarc::driver::CudaDevice;

    #[test]
    fn allocate_some_page_locked_mem() {
        let _device = CudaDevice::new(0).unwrap(); 
        {
            let mut vec = Vec::new_in(CudaHostAllocator);
            for _ in 0..10000 {
                vec.push(1);
            }
        }
        println!("Done");
    }
}
