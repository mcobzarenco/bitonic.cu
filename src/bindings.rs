#![allow(non_snake_case)]

use cudarc::driver::DeviceRepr;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

unsafe impl DeviceRepr for Struct {}

impl Default for Struct {
    fn default() -> Self {
        Self { value: 0.0 }
    }
}
