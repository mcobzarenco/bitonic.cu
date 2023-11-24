#ifndef STRUCT_H
#define STRUCT_H

// I use a struct here instead of just float to see how to generate
// bindings and create instances of `Struct` from rust on host.
struct Struct {
    float value;
};

#endif // STRUCT_H
