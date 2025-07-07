SetFactory("Built-in");
SetFactory("OpenCASCADE");
//+
Rectangle(1) = {-50, -50, 0, 100, 100, 0};
//+
Physical Curve("left", 1) = {4};
Physical Curve("right", 2) = {2};
Physical Curve("topbot", 3) = {3, 1};
