

SetFactory("OpenCASCADE");


a = 1; // x-radius
b = 5; // y-radius
c = 1; // z-radius


// Create a unit sphere
Sphere(1) = {0, 0, 0, 1};


// Scale sphere into an ellipsoid
Dilate {{0, 0, 0}, {a, b, c}} { Volume{1}; }

