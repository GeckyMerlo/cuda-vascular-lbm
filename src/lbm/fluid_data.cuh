struct FluidData {
public:
    double* density;
    double* velocity_x;
    double* velocity_y;
    double* velocity_z;
    double *f; // distribution functions
    double *f_temp; // temporary distribution functions for streaming step
    int nx, ny, nz;

    FluidData(int nx, int ny, int nz){
        cudaMalloc(&density, nx*ny*nz*sizeof(double));
        cudaMalloc(&velocity_x, nx*ny*nz*sizeof(double));
        cudaMalloc(&velocity_y, nx*ny*nz*sizeof(double));
        cudaMalloc(&velocity_z, nx*ny*nz*sizeof(double));
        cudaMalloc(&f, nx*ny*nz*19*sizeof(double)); 
        cudaMalloc(&f_temp, nx*ny*nz*19*sizeof(double));
        this->nx = nx;
        this->ny = ny;
        this->nz = nz;
    };
    
    ~FluidData(){
        cudaFree(density);
        cudaFree(velocity_x);
        cudaFree(velocity_y);
        cudaFree(velocity_z);
        cudaFree(f);
        cudaFree(f_temp);
    };
};