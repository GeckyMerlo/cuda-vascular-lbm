SetFactory("OpenCASCADE");

radius = 3.0;
length = 60.0;
lc = 0.8;

stenosis = 0.30;
rmin = radius * (1.0 - stenosis);

z0 = length/2;
stenosis_len = 12.0;

eps = 1e-6;

// asse + profilo parete
Point(1) = {0,      0, 0, lc};
Point(2) = {radius, 0, 0, lc};

Point(3) = {radius, 0, z0 - stenosis_len/2, lc};
Point(4) = {rmin,   0, z0, lc};
Point(5) = {radius, 0, z0 + stenosis_len/2, lc};

Point(6) = {radius, 0, length, lc};
Point(7) = {0,      0, length, lc};

// bordo esterno del vaso
Spline(1) = {2,3,4,5,6};

// chiusura sul piano x-z usando l'asse
Line(2) = {6,7}; // outlet radius
Line(3) = {7,1}; // axis
Line(4) = {1,2}; // inlet radius

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

// rivoluzione attorno a z
out[] = Extrude {{0,0,1}, {0,0,0}, 2*Pi} {
  Surface{1};
};

Coherence;

Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

inlet[] = Surface In BoundingBox{-radius-eps, -radius-eps, -eps,
                                  radius+eps,  radius+eps,  eps};

outlet[] = Surface In BoundingBox{-radius-eps, -radius-eps, length-eps,
                                   radius+eps,  radius+eps, length+eps};

all_surfaces[] = Boundary{ Volume{out[1]}; };
wall[] = all_surfaces[];
wall[] -= inlet[];
wall[] -= outlet[];

Physical Surface("INLET") = {inlet[]};
Physical Surface("OUTLET") = {outlet[]};
Physical Surface("WALL") = {wall[]};
Physical Volume("FLUID") = {out[1]};

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 1;
Mesh.Format = 1;