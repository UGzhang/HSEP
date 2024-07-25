#include "Atom.cuh"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

template <typename T>
static T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}
void expertVTK(const char* filename, Atom* atom);
void exportVTKXML(const char* filename, Atom* atom);