# Define the compiler and flags
CC = nvcc
CFLAGS = -g -G -O3 -arch=sm_86

# Define the target executable
TARGET = MolecularDynamics

# Define the build directory
BUILD_DIR = build
OUTPUT_DIR = output

# Define the source files
SRCS = quaternions.cu utils.cu surface.cu integrate.cu initialize.cu neighbour.cu force.cu timestep.cu main.cu

# Define the object files
OBJS = $(patsubst %.cu, $(BUILD_DIR)/%.o, $(SRCS))

# Define the executable
EXEC = ../$(TARGET)

# Default target
all: $(BUILD_DIR) $(OUTPUT_DIR) $(EXEC)

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Create the output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Link the object files to create the executable
$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lcudart -lstdc++

# Compile source files to object files
$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(CC) $(CFLAGS) -dc $< -o $@

# Linking device code separately
$(BUILD_DIR)/link.o: $(OBJS)
	$(CC) $(CFLAGS) -dlink $^ -o $@

# Clean up
clean:
	rm -rf $(BUILD_DIR) $(OUTPUT_DIR) $(EXEC)

.PHONY: all clean
