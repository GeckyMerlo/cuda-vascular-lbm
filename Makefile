# Build and run helper for the CUDA vascular LBM simulation.
# Edit the variables in this file before running `make run`.

CMAKE ?= cmake
BUILD_DIR ?= build-cuda
BUILD_TYPE ?= Release
GENERATOR ?= Ninja
CUDA_ARCHS ?= 60;70;75;80;86

TARGET := vascular_lbm

ifeq ($(OS),Windows_NT)
EXE := $(BUILD_DIR)/$(TARGET).exe
else
EXE := $(BUILD_DIR)/$(TARGET)
endif

# Positional simulation arguments.
MESH_FILE := msh/voxel_domain.bin
STEPS := 300
OUTPUT_INTERVAL := 20
TAU := 0.8

# Particle simulation arguments. Set MAX_PARTICLES to 0 to disable particles.
MAX_PARTICLES := 5000
RBC_RATE := 2
PLATELET_RATE := 0.2
LEUKOCYTE_RATE := 0.02
PARTICLE_OUTPUT_INTERVAL := $(OUTPUT_INTERVAL)

# Particle interaction parameters.
CONTACT_STIFFNESS := 0.05
CONTACT_DAMPING := 0.02
FRICTION := 0.2
WALL_STIFFNESS := 0.08
WALL_DAMPING := 0.02

SIM_ARGS := \
	$(MESH_FILE) \
	$(STEPS) \
	$(OUTPUT_INTERVAL) \
	$(TAU) \
	--max-particles $(MAX_PARTICLES) \
	--rbc-rate $(RBC_RATE) \
	--platelet-rate $(PLATELET_RATE) \
	--leukocyte-rate $(LEUKOCYTE_RATE) \
	--particle-output-interval $(PARTICLE_OUTPUT_INTERVAL) \
	--contact-stiffness $(CONTACT_STIFFNESS) \
	--contact-damping $(CONTACT_DAMPING) \
	--friction $(FRICTION) \
	--wall-stiffness $(WALL_STIFFNESS) \
	--wall-damping $(WALL_DAMPING)

FLUID_ARGS := $(MESH_FILE) $(STEPS) $(OUTPUT_INTERVAL) $(TAU)

.PHONY: all configure build run run-fluid clean clean-output print-args

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -G "$(GENERATOR)" -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_CUDA_ARCHITECTURES="$(CUDA_ARCHS)"

build: configure
	$(CMAKE) --build $(BUILD_DIR) --config $(BUILD_TYPE)

run: build
	$(EXE) $(SIM_ARGS)

run-fluid: build
	$(EXE) $(FLUID_ARGS)

print-args:
	@echo $(EXE) $(SIM_ARGS)

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)

clean-output:
	$(CMAKE) -E rm -rf output
