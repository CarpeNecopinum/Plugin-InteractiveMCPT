#include "InteractiveMCPT_.hh"

#include <OpenFlipper/BasePlugin/PluginFunctions.hh>
#include <OpenFlipper/BasePlugin/PluginFunctionsViewControls.hh>
#include <ObjectTypes/Sphere/Sphere.hh>
#include <ACG/Utils/VSToolsT.hh>

#include <QTime>

#define EPS (1e-6)



void InteractiveMCPTPlugin::initializePlugin()
{
	// Create the toolbox
	QWidget* toolbox = new QWidget();

	QVBoxLayout* toolboxLayout = new QVBoxLayout(toolbox);

    QPushButton* raytraceButtonClassic = new QPushButton("Open Interactive MCPT");
    connect(raytraceButtonClassic,SIGNAL(clicked()),this,SLOT(openWindow()));
	toolboxLayout->addWidget(raytraceButtonClassic);

	// Add the Toolbar to OpenFlipper Core
    emit addToolbox("Raytracer", toolbox, 0);

    imageWindow = new QWidget(0);
    imageWindow->setWindowTitle("InteractiveMCPT");

    QHBoxLayout* layout = new QHBoxLayout(imageWindow);
    imageWindow->setLayout(layout);
    imageWindow->resize(800, 600);

	imageLabel_ = new QLabel();
	imageLabel_->setBackgroundRole(QPalette::Base);
	imageLabel_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	imageLabel_->setScaledContents(false);
	imageLabel_->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(imageLabel_,SIGNAL(customContextMenuRequested(QPoint)),this,SLOT(showContextMenu(QPoint)));
    layout->addWidget(imageLabel_);

    QVBoxLayout* sidebox = new QVBoxLayout(imageWindow);
    layout->addLayout(sidebox);

    QPushButton* globalRenderButton = new QPushButton("FullImage MCPT",imageWindow);
    sidebox->addWidget(globalRenderButton);

    QGridLayout * sideboxGrid = new QGridLayout(imageWindow);
    sidebox->addLayout(sideboxGrid);
    sideboxGrid->addWidget(new QLabel("Rays per Pixel", imageWindow), 0, 0);

    QSpinBox * raysPerPixel = new QSpinBox(imageWindow);
    raysPerPixel->setMaximum(64);
    raysPerPixel->setMinimum(1);
    sideboxGrid->addWidget(raysPerPixel, 0, 1);

}

void InteractiveMCPTPlugin::showContextMenu(QPoint _point) {
    QMenu* menu = new QMenu(imageWindow);
	menu->addAction("Save Image",this,SLOT(saveImage()));
    menu->popup(imageWindow->mapToGlobal(_point));
}

void InteractiveMCPTPlugin::saveImage() {
	// Ask user for filename
    QString fileName = QFileDialog::getSaveFileName(imageWindow,"Save image to",OpenFlipperSettings().value("Core/CurrentDir").toString()+QDir::separator()+"raytracedImage.png","Images (*.png *.xpm *.jpg)" );

	// Save the image
	if (fileName != "" )
		image_.save(fileName);
}

void InteractiveMCPTPlugin::openWindow() {
    cancel_ = false;

    clearImage();

    imageLabel_->resize(PluginFunctions::viewerProperties().glState().viewport_width(),PluginFunctions::viewerProperties().glState().viewport_height());
    updateImageWidget();
    imageWindow->show();
}

void InteractiveMCPTPlugin::launchThread() {

	cancel_ = false;

    OpenFlipperThread* tread = new OpenFlipperThread("MCPT Thread");

	// Connect the appropriate signals
	connect(tread, SIGNAL(function()), this, SLOT(raytrace()), Qt::DirectConnection);

	// Connect the appropriate signals
	connect(tread, SIGNAL(finished(QString)), this, SIGNAL(finishJob(QString)), Qt::DirectConnection);
	connect(tread, SIGNAL(finished(QString)), this, SLOT(threadFinished()), Qt::DirectConnection);

	// Calculate number of iterations for progress indicator
	int maxIterations = PluginFunctions::viewerProperties().glState().viewport_width() * PluginFunctions::viewerProperties().glState().viewport_height();

	// Tell core about my thread
	// Note: The last parameter determines whether the thread should be blocking
	emit startJob( "RayTracingThread", "RayTracing" , 0 , maxIterations , false);

	// Start internal QThread
	tread->start();

	// Start actual processing of job
	tread->startProcessing();

	// Update widget every second
	updateTimer_.setInterval(1000);
	updateTimer_.setSingleShot(false);
	connect(&updateTimer_,SIGNAL(timeout()),this,SLOT(updateImageWidget()) );
	updateTimer_.start();

	// Resize the widget to the image and show it
	imageLabel_->resize(PluginFunctions::viewerProperties().glState().viewport_width(),PluginFunctions::viewerProperties().glState().viewport_height());

	// First redraw to initialize the widget
	updateImageWidget();
    imageWindow->show();
}

void InteractiveMCPTPlugin::canceledJob(QString /*_jobId*/ ) {
	emit log(LOGERR, "Cancel Button");
	cancel_ = true;
}

void InteractiveMCPTPlugin::threadFinished() {

	// Stop the update timer
	updateTimer_.stop();
	updateTimer_.disconnect();

	// Last update of image
	updateImageWidget();

}

void InteractiveMCPTPlugin::updateImageWidget() {

	// update the widget
	if ( ! cancel_ ) {
		imageLabel_->setPixmap(QPixmap::fromImage(image_));
		imageLabel_->resize(image_.size());
        imageWindow->adjustSize();
	}

}

/** \brief Main loop of the raytracer
 *
 * This function contains the main loop of the raytracer and the setup of the shot rays
 */
void InteractiveMCPTPlugin::raytrace() {
	QColor        pixelColor(0,0,0);                 // Variable for color transformation to QImage (don't use!)
	Vec3d         viewingDirection;                  // Viewing direction (world coordinates)
	Vec3d         x_dir, y_dir;                      // In scene direction vectors (parallel to image plane) (world coordinates)
	Vec3d         start_point;                       // Center of the Image Plane (world coordinates)
	Vec3d         current_point;                     // current rendering point on the image plane (world coordinates)
	double        imagePlanewidth, imagePlaneHeight; // Size of the image plane (world coordinates)
	double        focalDistance;                     // distance eyepos to image plane
	double        fovy;                              // Field of view in y direction
	double        aspect;                            // Aspect ratio of the image
	double        imageWidth,imageHeight;            // Height and width of the image;
	Ray           ray;                               // Ray shot into the scene
	Color         col;                               // color returned for current ray

	QTime	execTime;
	execTime.start();


	// Get field of view in y direction from the viewer
	fovy = PluginFunctions::viewerProperties().glState().fovy();

	// Get the viewing direction and aspect ratio
	viewingDirection = PluginFunctions::viewingDirection();
	aspect           = PluginFunctions::viewerProperties().glState().aspect();

	// Set the distance to the image plane to 1.0 (This value does not really matter as we only need the direction)
	focalDistance = 1.0;

	// Width & height of the image plane in world coordinates
	imagePlaneHeight = 2.0 * focalDistance * tan( 0.5 * fovy );
	imagePlanewidth  = imagePlaneHeight * aspect;

	// setup direction vectors at the image plane
	// These vectors are parallel to image plane in world coordinates and have the length of one pixel
	// If you go one pixel up in the image, you have to go one step into the direction of the computed vector
	// to get the corresponding point on the image plane in world coordinates.
	x_dir = ( viewingDirection % PluginFunctions::upVector() ).normalize() * imagePlanewidth / imageWidth;
	y_dir = ( x_dir % viewingDirection).normalize() * imagePlaneHeight / imageHeight;

	// Get the start point on the image plane.
	// The rays start at the eye pos and go into the direction of the point on the image plane.
	start_point = PluginFunctions::eyePos() + viewingDirection - 0.5f * imageWidth * x_dir + 0.5f * imageHeight * y_dir;
	ray.origin  = PluginFunctions::eyePos();

	// Collect all light sources in the scene
	lights_.clear();
	for ( PluginFunctions::ObjectIterator o_It(PluginFunctions::TARGET_OBJECTS, DataType( DATA_LIGHT) ) ; o_It != PluginFunctions::objectsEnd(); ++o_It) {
		// Get the associated light source data
		LightSource* light = PluginFunctions::lightSource(*o_It);
		if ( !light )
			continue;

		lights_.push_back(*light);
	}

	// Raytrace the image (Iterate over all pixels and shoot rays into scene)
	for (unsigned int y = 0; y < imageHeight; ++y)
	{
		for (unsigned int x = 0; x < imageWidth; ++x)
		{
			// Cancel processing if requested by user
			if ( cancel_ ) {
				return;
			}

			// Set the current iteration in the progress
			emit setJobState("RayTracingThread",y * imageWidth + x);

			// compute the current point on the image plane
			current_point = start_point + x_dir * x - y_dir * y;

			// setup current ray direction
			ray.direction = (current_point - ray.origin).normalize();

			// trace the ray through the scene, get its color
			col = trace(ray, 0);

			// clamp color values
			col.minimize(Color(1.0, 1.0, 1.0, 1.0));
			col.maximize(Color(0.0, 0.0, 0.0, 1.0));

			// Set the returned pixel color in the image
			pixelColor.setRgbF(col[0],col[1],col[2]);
			image_.setPixel( QPoint(x,y) , pixelColor.rgb() );
		}
	}

	std::cout << "RayTracing: " << (double)execTime.elapsed() / 1000 << " seconds" << std::endl;

}

/** \brief compute color from ray
 *
 * This function shoots a ray into the scene and computes the visible color along this ray.
 * The _depth parameter restricts the number of recursions for mirrored rays.
 */
Color InteractiveMCPTPlugin::trace(const Ray& _ray, unsigned int _depth) {

	// Do at most 3 recursive steps for mirorred contributions
	unsigned int max_depth = 3;

	// Set background color to black
	Color background(0.0f,0.0f,0.0f,1.0f);

	// Stop if maximum recursion depth is reached
	if (_depth > max_depth)
		return background;

	Vec3d          point, normal;              // Temporary variables for data at ray/object intersection point
	double         parameter;                  // Temporary variable for parametric depth of intersection point (distance from current point to intersection point )

	Vec3d          intersectionPoint;          // Intersection between ray and object
	Vec3d          intersectionNormal;         // Normal at intersection point
	double         intersectionDepth(FLT_MAX); // parametric depth along intersection ray (initialized to max as we search for the closest intersection)
	TriMeshObject* intersectedObject(0);       // last intersected Object
	Material       objectMaterial;             // Material of last intersected object

	// Iterate over all meshes in the scene and check if they get intersected
	// The intersection point/normal, parameter and object are initialized here
	for(PluginFunctions::ObjectIterator o_It( PluginFunctions::ALL_OBJECTS, DataType( DATA_TRIANGLE_MESH | DATA_SPHERE) ) ; o_It != PluginFunctions::objectsEnd(); ++o_It) {

		// Intersect current ray with the current mesh/sphere and get normal, color, and point of intersection. Additionally check if the ray
		// intersects the current object earlier than any other object.
		if (intersect(*( *o_It), _ray, point, normal, parameter) && parameter < intersectionDepth )
		{
			intersectionDepth  = parameter;
			objectMaterial     = o_It->materialNode()->material();
			intersectionPoint  = point;
			intersectionNormal = normal;
		}

	}

	// If no intersection is found -> return background color
	if (intersectionDepth == FLT_MAX)
		return background;

	// backfaces are black
	if( (intersectionNormal | (-_ray.direction)) < 0.0 )
		return background;

	// Compute the reflected ray at the intersection point (This function has to be implemented in the exercise. See below)
	Ray reflectedRay(reflect(_ray, intersectionPoint, intersectionNormal));

	// ==================================================================
	// These variables have to be computed in the exercise
	// ==================================================================

	Ray            lightRay;                             // Use this variable to set up the ray for shadow testing
	bool           inShadow;                             // Use this variable to mark if a point is in shadow
	double         dot;                                  // dot product between Light direction and intersection normal
	Color          reflectedColor(0.0f,0.0f,0.0f,1.0f);  // Sum up the different components of the reflected color in this varialble
	Color          mirroredColor(0.0f,0.0f,0.0f,1.0f);   // Sum up the mirrored color in this variable

	// Compute reflected contribution of each light source
	for ( unsigned int i = 0 ; i < lights_.size() ; ++i ) {

		LightSource& light = lights_[i];

		// Compute the direction of the incoming light and the distance from the intersection point to the light source
		Vec3d  lightDirection = (light.position() - intersectionPoint );
		double lightDistance  = lightDirection.norm();

		// Normalize light direction vector
		lightDirection.normalize();

			// INSERT CODE
			// reflected contribution of each light source:
			//    point in shadow?
			//    local lighting: ambient, diffuse, specular term
			//
			// Access to variables:
			//
			//   light.position()
			//   light.ambientColor()
			//   light.diffuseColor()
			//   light.specularColor()
			//
			//   objectMaterial.ambientColor()
			//   objectMaterial.diffuseColor()
			//   objectMaterial.specularColor()
			//   objectMaterial.shininess()
			//
			//   Intersection with all objects: see ObjectIterator above
			//
			//--- start strip ---

            lightRay.direction = lightDirection.normalized();
            lightRay.origin = intersectionPoint;

            // Shadow test
            inShadow = false;
            for(PluginFunctions::ObjectIterator o_It( PluginFunctions::ALL_OBJECTS, DataType( DATA_TRIANGLE_MESH | DATA_SPHERE) ) ; o_It != PluginFunctions::objectsEnd(); ++o_It) {
                Vec3d dummyPoint, dummyNormal;
                double occluderDistance;
                if (intersect(*( *o_It), lightRay, dummyPoint, dummyNormal, occluderDistance)  && (occluderDistance < lightDistance))
                {
                    inShadow = true;
                    break;
                }
            }

            // If the ray comes from the back, the face shadows itself
            if ((lightRay.direction | normal) < 0.0) inShadow = true;

            // Ambient Term, regardless of shadowing
            reflectedColor += light.ambientColor() * objectMaterial.ambientColor();

            // Diffuse and specular term, only if seen by lightsource, no attenuation by distance
            if (!inShadow)
            {
                // Diffuse Term
                double diffuseFactor = dot = (lightDirection | normal);
                reflectedColor += (light.diffuseColor() * objectMaterial.diffuseColor()) * diffuseFactor;

                // Specular Term
                double specularFactor = std::pow(reflectedRay.direction | lightRay.direction, objectMaterial.shininess());
                reflectedColor += (light.specularColor() * objectMaterial.specularColor()) * specularFactor;
            }


			//--- end strip ---
	}


	// INSERT CODE
	// mirror effects: recursive trace()
	//
	// objectMaterial.reflectance()
	//--- start strip ---

    mirroredColor = trace(reflectedRay, _depth + 1);


	//--- end strip ---

	double r = objectMaterial.reflectance();
	return (reflectedColor * (1.0 - r) + mirroredColor * r);

}

/** \brief Intersect ray with  object
 *
 * This function intersects an object in the scene with a ray and returns the intersection point, the intersection normal and the distance to the intersection point
 */
bool InteractiveMCPTPlugin::intersect(BaseObjectData&      _object,
		const Ray&           _ray,
		Vec3d&               _intersection,
		Vec3d&               _normal,
		double&              _t )
{
	_t = FLT_MAX;

	Vec3d  ip;
	double t;

	if ( PluginFunctions::triMeshObject(&_object)  ) {

		ACG::Vec3d bb_min,bb_max;
		PluginFunctions::triMeshObject(&_object)->boundingBox(bb_min,bb_max);

		if ( !intersectBoundingBox(bb_min,bb_max,_ray) )
			return false;

		TriMesh& mesh = *( PluginFunctions::triMeshObject(&_object)->mesh() );

		TriMesh::FaceIter f_it( mesh.faces_begin() );
		TriMesh::FaceIter f_end( mesh.faces_end() );

		for (; f_it!=f_end; ++f_it)
		{
		  Face face(mesh,*f_it);

		  if (intersectTriangle( face, mesh.normal(*f_it), _ray, ip, t) && (t < _t))
		  {
		    _t            = t;
		    _intersection = ip;
		    _normal       = mesh.normal(*f_it);
		  }
		}
		//TriMeshObject* mesh_object = PluginFunctions::triMeshObject(&_object);
		//TriMesh& mesh = *(mesh_object->mesh() );

		//TriMeshObject::OMTriangleBSP* bsp = mesh_object->requestTriangleBsp();

		//TriMeshObject::OMTriangleBSP::RayCollision rc = bsp->nearestRaycollision(_ray.origin + EPS*_ray.direction, _ray.direction);
		//TriMeshObject::OMTriangleBSP::RayCollision rc = bsp->directionalRaycollision(_ray.origin + EPS*_ray.direction, _ray.direction);

		//if(!rc.empty()) {
		//	TriMesh::FaceHandle handle = rc.front().first;
		//	_t = rc.front().second - EPS;
		//	_normal = mesh.normal(handle);
		//	_intersection = _ray.origin + _t * _ray.direction;
		//}

	} else  if ( PluginFunctions::sphereObject(&_object)  ) {
		SphereNode* sphere = PluginFunctions::sphereNode(&_object);

		ACG::Vec3d normal(1.0,1.0,1.0);

		if (intersectSphere( sphere->get_primitive(0).position, sphere->get_primitive(0).size , _ray, ip, normal, t) && (t < _t))
		{
			_t            = t;
			_intersection = ip;
			_normal       = normal;
		}

	}

	return (_t != FLT_MAX);
}

/** \brief Compute a reflection ray
 *
 * This function reflects the given ray at the point _point with the normal _normal.
 */
Ray
InteractiveMCPTPlugin::reflect(const Ray&   _ray,
		const Vec3d& _point,
		const Vec3d& _normal) {
	Ray result;

	// INSERT CODE
	// mirror _ray and return the normalized result in result
	//--- start strip ---


    Vec3d normal = _normal.normalized();
    Vec3d dir    = _ray.direction.normalized();

    Vec3d mirrored = dir - normal * 2.0 * (normal | dir);

    result.direction = mirrored;
    result.origin = _point;
	//--- end strip ---

	return result;
}

/** \brief Intersect a ray with a Triangle
 *
 * This function intersects a ray with a given face.
 * It returns true if the face has been intersected. Additionally it returns the intersection point,normal and
 * the intersection parameter.
 */
bool
InteractiveMCPTPlugin::intersectTriangle( const Face&        _face,
		const Vec3d&       _normal,
		const Ray&         _ray,
		Vec3d&             _intersection,
		double&            _t ) const
{
	// INSERT CODE
	// intersect _ray with _face with _normal
	// store intersection point in _intersection
	// store ray parameter in _t
	// return true if intersection occurs AND t > EPS
	//
	// Hint: Use the barycentric coordinate function : bary_coord(_intersection, _face.p0, _face.p1, _face.p2, bary);
	//--- start strip ---
        
    // Project _ray.origin into the Triangle's plane as _intersection

    _t = ((_face.p0 - _ray.origin) | _normal) / (_normal | _ray.direction);
    if (_t <= EPS) return false;

    _intersection = _ray.origin + _t * _ray.direction;

    Vec3d bary;
    if (bary_coord(_intersection, _face.p0, _face.p1, _face.p2, bary))
        if (bary[0] > 0. && bary[1] > 0. && bary[2] > 0.)
            return true;

	//--- end strip ---

	return false;
}

/** \brief Intersect a ray with a Sphere
 *
 * This function intersects a ray with a given sphere.
 * It returns true if the sphere has been intersected. Additionally it returns the intersection point, normal and
 * the intersection parameter.
 */
bool InteractiveMCPTPlugin::intersectSphere( const Vec3d&      _center,
		const double&     _radius,
		const Ray&        _ray,
		Vec3d&            _intersection,
		Vec3d&            _normal,
		double&           _t ) const
{
	// INSERT CODE
	// intersect _ray with sphere ( at _center with radius _radius)
	// store intersection point in _intersection
	// store normal at intersection point in _normal
	// store ray parameter in _t
	// return true if intersection occurs AND t > EPS
	//--- start strip ---

    //Project the ray origin onto the normal plane around the sphere center
    _t = (_center - _ray.origin) | _ray.direction; // Temporary _t to the plane
    Vec3d relative = _center - (_ray.origin + _ray.direction * _t);

    double radiusSq = _radius * _radius;
    double relativeSq = relative | relative;

    // If the ray hits the tangent plane further than radius from the center,
    // it doesn't hit the sphere either
    if (radiusSq < relativeSq) return false;

    // Go the "height" of the sphere back
    _t -= std::sqrt(_radius * _radius - (relative | relative));
    _intersection = _ray.origin + _ray.direction * _t;
    _normal = (_intersection - _center) / _radius;

    if (_t >= EPS) return true;

	//--- end strip ---
	return false;
}

/** \brief Compute barycentric coordinates
 *
 * This function computes the barycentric coordinates of the point p in the triangle
 * u,v,w and returns the result in _result.
 */
bool InteractiveMCPTPlugin::bary_coord(  const Vec3d &  _p,
		const Vec3d &  _u,
		const Vec3d &  _v,
		const Vec3d &  _w,
		Vec3d &        _result ) const
{
	Vec3d  vu = _v - _u,
	wu = _w - _u,
	pu = _p - _u;

	// find largest absolute coordinate of normal
	double nx = vu[1] * wu[2] - vu[2] * wu[1],
				 ny        = vu[2] * wu[0] - vu[0] * wu[2],
				 nz        = vu[0] * wu[1] - vu[1] * wu[0],
				 ax        = fabs(nx),
				 ay        = fabs(ny),
				 az        = fabs(nz);

	unsigned char max_coord;

	if ( ax > ay ) {
		if ( ax > az ) {
			max_coord = 0;
		}
		else {
			max_coord = 2;
		}
	}
	else {
		if ( ay > az ) {
			max_coord = 1;
		}
		else {
			max_coord = 2;
		}
	}

	// solve 2D problem
	switch (max_coord)
	{
		case 0:
			{
				if (1.0+ax == 1.0) return false;
				_result[1] = 1.0 + (pu[1]*wu[2]-pu[2]*wu[1])/nx - 1.0;
				_result[2] = 1.0 + (vu[1]*pu[2]-vu[2]*pu[1])/nx - 1.0;
				_result[0] = 1.0 - _result[1] - _result[2];
			}
			break;

		case 1:
			{
				if (1.0+ay == 1.0) return false;
				_result[1] = 1.0 + (pu[2]*wu[0]-pu[0]*wu[2])/ny - 1.0;
				_result[2] = 1.0 + (vu[2]*pu[0]-vu[0]*pu[2])/ny - 1.0;
				_result[0] = 1.0 - _result[1] - _result[2];
			}
			break;

		case 2:
			{
				if (1.0+az == 1.0) return false;
				_result[1] = 1.0 + (pu[0]*wu[1]-pu[1]*wu[0])/nz - 1.0;
				_result[2] = 1.0 + (vu[0]*pu[1]-vu[1]*pu[0])/nz - 1.0;
				_result[0] = 1.0 - _result[1] - _result[2];
			}
			break;
	}

    return true;
}

void InteractiveMCPTPlugin::clearImage()
{
    // Get the width and height of the current viewport
    int imageWidth  = PluginFunctions::viewerProperties().glState().viewport_width();
    int imageHeight = PluginFunctions::viewerProperties().glState().viewport_height();

    // Create a QImage of the viewer size and clear it
    image_ = QImage(imageWidth,imageHeight,QImage::Format_RGB32);
    image_.fill(Qt::black);
}


bool InteractiveMCPTPlugin::intersectBoundingBox(const Vec3d& bb_min , const Vec3d& bb_max ,const Ray& _ray){

	double t_near=-DBL_MAX, t_far=DBL_MAX, t1, t2;

	for (int i=0; i<3; ++i)
	{
		if (1.0+_ray.direction[i] == 1.0)
		{
			if (_ray.origin[i] < bb_min[i] || _ray.origin[i] > bb_max[i]) {
				return false;
			}
		}
		else
		{
			t1 = (bb_min[i] - _ray.origin[i]) / _ray.direction[i];
			t2 = (bb_max[i] - _ray.origin[i]) / _ray.direction[i];
			if (t1 > t2)      std::swap(t1, t2);
			if (t1 > t_near)  t_near = t1;
			if (t2 < t_far)   t_far  = t2;
			if (t_near > t_far || t_far < 0) {
				return false;
			}
		}
	}

	return true;
}


#if QT_VERSION < 0x050000
  Q_EXPORT_PLUGIN2( interactiveMCPTplugin , InteractiveMCPT );
#endif
