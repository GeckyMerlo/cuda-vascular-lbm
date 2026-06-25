// vena_cilindrica.geo
// Cilindro cavo semplificato per test LBM/voxelizzazione
// Asse del vaso lungo Z
// Physical Surface: INLET, OUTLET, WALL
// Physical Volume: FLUID

SetFactory("OpenCASCADE");

// -------------------------
// Parametri
// -------------------------
radius = 3.0;      // raggio interno [mm]
length = 30.0;     // lunghezza [mm]
lc = 0.8;          // dimensione caratteristica mesh [mm]

// -------------------------
// Geometria: cilindro pieno che rappresenta il dominio fluido interno
// Base al piano z=0, uscita al piano z=length
// -------------------------
Cylinder(1) = {0, 0, 0, 0, 0, length, radius, 2*Pi};

// Sincronizza OpenCASCADE con il modello Gmsh
Coherence;

// -------------------------
// Mesh size globale
// -------------------------
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

// -------------------------
// Recupero superfici tramite bounding box
// Nota: con OpenCASCADE il cilindro ha:
// - una superficie laterale
// - una faccia a z=0
// - una faccia a z=length
// Usiamo bounding box per identificarle in modo robusto.
// -------------------------
eps = 1e-6;

inlet[] = Surface In BoundingBox{-radius-eps, -radius-eps, -eps,
                                  radius+eps,  radius+eps,  eps};

outlet[] = Surface In BoundingBox{-radius-eps, -radius-eps, length-eps,
                                   radius+eps,  radius+eps, length+eps};

all_surfaces[] = Boundary{ Volume{1}; };
wall[] = all_surfaces[];

// Rimuove inlet e outlet dalla lista wall
wall[] -= inlet[];
wall[] -= outlet[];

// -------------------------
// Physical groups utili allo script Python
// -------------------------
Physical Surface("INLET") = {inlet[]};
Physical Surface("OUTLET") = {outlet[]};
Physical Surface("WALL") = {wall[]};
Physical Volume("FLUID") = {1};

// -------------------------
// Opzioni mesh
// -------------------------
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
Mesh.Format = 1; // ASCII .msh
