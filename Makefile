# Build and run helper for the CUDA vascular LBM simulation.
# Edit the variables in this file before running `make run`.

NVCC ?= nvcc
BUILD_DIR ?= build-cuda
OBJ_DIR := $(BUILD_DIR)/obj
BUILD_TYPE ?= Release
CUDA_ARCHS ?= 75 80 86

TARGET := vascular_lbm

ifeq ($(OS),Windows_NT)
EXE := $(BUILD_DIR)/$(TARGET).exe
else
EXE := $(BUILD_DIR)/$(TARGET)
endif

SOURCES := \
	src/main.cu \
	src/lbm/lbm_constants.cu \
	src/lbm/lbm_system.cpp \
	src/space/space_system.cpp \
	src/particles/particle_system.cpp \
	src/forces/force_model.cpp \
	src/msh_utils/mesh_reader.cpp

OBJECTS := $(SOURCES:%=$(OBJ_DIR)/%.o)
DEPS := $(OBJECTS:.o=.d)

CUDA_ARCH_LIST := $(subst ;, ,$(CUDA_ARCHS))
GENCODE_FLAGS := $(foreach arch,$(CUDA_ARCH_LIST),-gencode arch=compute_$(arch),code=sm_$(arch))

INCLUDES := -Isrc
CPPFLAGS ?=
NVCCFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

ifeq ($(BUILD_TYPE),Debug)
OPT_FLAGS ?= -O0 -g -G
else
OPT_FLAGS ?= -O3 -DNDEBUG
endif

COMMON_NVCC_FLAGS := --std=c++17 -rdc=true $(OPT_FLAGS) $(GENCODE_FLAGS) $(INCLUDES) $(CPPFLAGS) $(NVCCFLAGS)
DEPFLAGS := -MMD -MP

# Positional and runtime simulation arguments.
MESH_FILE := msh/cilindric_vessel_stenosis30_voxel_domain.bin
# MESH_FILE := msh/voxel_domain.bin
STEPS := 30000
OUTPUT_INTERVAL := 300
WARMUP_STEPS := 20000
TAU := 0.8

# Particle simulation arguments. Set MAX_PARTICLES to 0 to disable particles.
MAX_PARTICLES := 10000
RBC_RATE := 0.01
PLATELET_RATE := 0.005
LEUKOCYTE_RATE := 0.005
PARTICLE_OUTPUT_INTERVAL := $(OUTPUT_INTERVAL)
PARTICLE_SUBSTEPS := 8

# Particle interaction parameters.
CONTACT_STIFFNESS := 0.02
CONTACT_DAMPING := 0.06
FRICTION := 0.2
WALL_STIFFNESS := 0.03
WALL_DAMPING := 0.06
MAX_PARTICLE_FORCE := 0.02
MAX_PARTICLE_SPEED := 0.06

SIM_ARGS := \
	$(MESH_FILE) \
	$(STEPS) \
	$(OUTPUT_INTERVAL) \
	$(TAU) \
	--warmup-steps $(WARMUP_STEPS) \
	--max-particles $(MAX_PARTICLES) \
	--rbc-rate $(RBC_RATE) \
	--platelet-rate $(PLATELET_RATE) \
	--leukocyte-rate $(LEUKOCYTE_RATE) \
	--particle-output-interval $(PARTICLE_OUTPUT_INTERVAL) \
	--particle-substeps $(PARTICLE_SUBSTEPS) \
	--contact-stiffness $(CONTACT_STIFFNESS) \
	--contact-damping $(CONTACT_DAMPING) \
	--friction $(FRICTION) \
	--wall-stiffness $(WALL_STIFFNESS) \
	--wall-damping $(WALL_DAMPING) \
	--max-particle-force $(MAX_PARTICLE_FORCE) \
	--max-particle-speed $(MAX_PARTICLE_SPEED)

FLUID_ARGS := $(MESH_FILE) $(STEPS) $(OUTPUT_INTERVAL) $(TAU) --warmup-steps $(WARMUP_STEPS)

.PHONY: all build run run-fluid clean clean-output print-args print-build

all: build

build: $(EXE)

$(EXE): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(NVCC) $(COMMON_NVCC_FLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(OBJ_DIR)/%.o: %
	@mkdir -p $(dir $@)
	$(NVCC) $(COMMON_NVCC_FLAGS) $(DEPFLAGS) -x cu -dc $< -o $@

run: build
	$(EXE) $(SIM_ARGS)

run-fluid: build
	$(EXE) $(FLUID_ARGS)

print-args:
	@echo $(EXE) $(SIM_ARGS)

print-build:
	@echo NVCC=$(NVCC)
	@echo BUILD_TYPE=$(BUILD_TYPE)
	@echo CUDA_ARCHS=$(CUDA_ARCH_LIST)
	@echo EXE=$(EXE)

clean:
	rm -rf $(BUILD_DIR)

clean-output:
	rm -rf output

-include $(DEPS)
