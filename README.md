
### Discrete Element Method
This is a mini project to simulate particle movement for the lecture 'High End Simulation in Practice' at FAU. It can run on Intel A40/A100 GPU.<br>
### Team members:<br>
Klöppner, Florian<br>
Zeynalli, Jahandar<br>
Zhang, Youyi<br>

![slice](Sphere.gif)

### WHAT WAS DONE (JUST THE BIG STUFF):<br>
1. Loading in complex geometry with stl files (From blender) and calculate correct particle wall interactions<br>
2. Implementation of Quaternion calculations<br>
3. Adaptive Time Stepping<br>
4. General Improvement of previous data structures and code<br>


### RUNNING THE CODE:
CODE IS WRITTEN FOR A40 GPU on alex<br>
1. Allocate interactive node on alex with a40<br>
2. Module load nvhpc cuda<br>
3. Execute run.sh<br>

### VISUALIZING RESULTS:
1. Copy output folder and used stl file to local machine <br>
2. load stl file in paraview and set opacity to 20% <br>
3. load output folder. Then set type to 3d glyphs, representation to sphere and scaling factor to 0.2 <br>

### HOW TO CONFIGURE CODE TO YOUR LIKING:
1. At the beginning of main.cu, all relevant system variables are set and can be adjusted. If wanted stl and vtk files for geometry/particles respectively can be set there as well.  
2. IMPORTANT: make sure domain size is large enough and stl file coordinates strictly positive, such that a cube from 0,0,0 to domainSize, domainSize, domainSize fully encapsulates that file. this cube defines the gridspace for the particles and is not automatically adjusted to new geometry  
3. ADDITIONALLY: Make sure all Particles defined in STL File are located inside the geometry and grid

GENERAL CODE STRUCTURE:  
1. Everything is started in the main function and then called for initialization/calculations from there.  
2. Classes are structured such that only thematically similar functions are in the same class.