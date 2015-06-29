#include "InteractiveMCPT_.hh"

#include <OpenFlipper/BasePlugin/PluginFunctions.hh>
#include <OpenFlipper/BasePlugin/PluginFunctionsViewControls.hh>
#include <ObjectTypes/Sphere/Sphere.hh>
#include <ACG/Utils/VSToolsT.hh>

#if QT_VERSION >= 0x050000
#include <QtConcurrent>
#endif

#include <QTime>
#include "Sampling.hh"
#include "BRDF.hh"
#include <QMouseEvent>
#include "ImageViewer.hh"

#define EPS (1e-6)


void InteractiveMCPTPlugin::testMousePressed(QMouseEvent *ev){
    std::cout << ev->pos().x() << " " << ev->pos().y() << std::endl;
}

void InteractiveMCPTPlugin::initializePlugin()
{
    mAccumulatedColor = 0;
    mSamples = 0;
    mQueuedSamples = 0;
    settings.samplesPerPixel = 1;

	// Create the toolbox
	QWidget* toolbox = new QWidget();

	QVBoxLayout* toolboxLayout = new QVBoxLayout(toolbox);

    QPushButton* raytraceButtonClassic = new QPushButton("Open Interactive MCPT");
    connect(raytraceButtonClassic,SIGNAL(clicked()),this,SLOT(openWindow()));
	toolboxLayout->addWidget(raytraceButtonClassic);

	// Add the Toolbar to OpenFlipper Core
    emit addToolbox("Interactive MCPT", toolbox, 0);

    imageWindow = new QWidget(0);
    imageWindow->setWindowTitle("InteractiveMCPT");

    QHBoxLayout* layout = new QHBoxLayout(imageWindow);
    imageWindow->setLayout(layout);
    imageWindow->resize(800, 600);

    imageLabel_ = new ImageViewer(imageWindow);
	imageLabel_->setBackgroundRole(QPalette::Base);
	imageLabel_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	imageLabel_->setScaledContents(false);
	imageLabel_->setContextMenuPolicy(Qt::CustomContextMenu);
	connect(imageLabel_,SIGNAL(customContextMenuRequested(QPoint)),this,SLOT(showContextMenu(QPoint)));
    connect(imageLabel_,SIGNAL(mousePressed(QMouseEvent*)),this,SLOT(testMousePressed(QMouseEvent*)));
    layout->addWidget(imageLabel_);

    QVBoxLayout* sidebox = new QVBoxLayout(imageWindow);
    layout->addLayout(sidebox);

    QPushButton* globalRenderButton = new QPushButton("FullImage MCPT",imageWindow);
    connect(globalRenderButton, SIGNAL(clicked()), this, SLOT(globalRender()));
    sidebox->addWidget(globalRenderButton);

    // Input: Rays per Pixel
    QGridLayout * sideboxGrid = new QGridLayout(imageWindow);
    sidebox->addLayout(sideboxGrid);
    sideboxGrid->addWidget(new QLabel("Rays per Pixel", imageWindow), 0, 0);

    QSpinBox * seRaysPerPixel = new QSpinBox(imageWindow);
    seRaysPerPixel->setMaximum(64);
    seRaysPerPixel->setMinimum(1);
    connect(seRaysPerPixel, SIGNAL(valueChanged(int)), this, SLOT(changeRaysPerPixel(int)));
    sideboxGrid->addWidget(seRaysPerPixel, 0, 1);


    connect(&updateTimer_,SIGNAL(timeout()),this,SLOT(updateImageWidget()) );
    updateTimer_.setInterval(1000);
    updateTimer_.setSingleShot(false);
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
    mCam = computeCameraInfo();
    imageWindow->show();

}


void InteractiveMCPTPlugin::runJob(RenderJob job)
{
    std::vector<Point>::iterator end = job.pixels.end();
    for (std::vector<Point>::iterator it = job.pixels.begin(); it != end; ++it)
    {
        Point& point = *it;

        for (int i = 0; i < job.settings.samplesPerPixel; i++)
            tracePixel(point.x, point.y);

        size_t index = point.y * image_.width() + point.x;
        mQueuedSamples[index]--;
    }

}

void InteractiveMCPTPlugin::tracePixel(size_t x, size_t y)
{
    /* randomly shift the ray direction for AA */
    double xd = x + double(std::rand()) / double(RAND_MAX) - 0.5;
    double yd = y + double(std::rand()) / double(RAND_MAX) - 0.5;

    /* Ray Setup */
    Vec3d current_point = mCam.image_plane_start + mCam.x_dir * xd - mCam.y_dir * yd;
    Ray ray = {mCam.eye_point, (current_point - mCam.eye_point).normalize()};

    /* Actual Path Tracing */

    Color color = trace(ray, 0);

    /* Write to accumulated Buffer + Sample counter */

    size_t index = x + y * image_.width();
    mAccumulatedColor[index] += Vec3d(color[0], color[1], color[2]);
    mSamples[index]++;
}

InteractiveMCPTPlugin::CameraInfo InteractiveMCPTPlugin::computeCameraInfo() const
{
    double fovy = PluginFunctions::viewerProperties().glState().fovy();
    Vec3d viewingDirection = PluginFunctions::viewingDirection();
    double aspect           = PluginFunctions::viewerProperties().glState().aspect();
    double focalDistance = 1.0;
    double imagePlaneHeight = 2.0 * focalDistance * tan( 0.5 * fovy );
    double imagePlanewidth  = imagePlaneHeight * aspect;
    Vec3d x_dir = ( viewingDirection % PluginFunctions::upVector() ).normalize() * imagePlanewidth / image_.width();
    Vec3d y_dir = ( x_dir % viewingDirection).normalize() * imagePlaneHeight / image_.height();
    Vec3d start_point = PluginFunctions::eyePos() + viewingDirection - 0.5 * image_.width() * x_dir + 0.5 * image_.height() * y_dir;

    CameraInfo cam = {x_dir, y_dir, start_point, PluginFunctions::eyePos()};
    return cam;
}

void InteractiveMCPTPlugin::queueJob(RenderJob job)
{
    for (Point point : job.pixels)
    {
        size_t index = point.y * image_.width() + point.x;
        mQueuedSamples[index]++;
    }
    mRunningFutures.push_back(QtConcurrent::run(this, &InteractiveMCPTPlugin::runJob, job));
}


void InteractiveMCPTPlugin::globalRender()
{
    RenderJob job;
    job.settings = settings;

    const int imageWidth  = image_.width();
    const int imageHeight = image_.height();
    for (int y = 0; y < imageHeight; ++y)
    {
        for (int x = 0; x < imageWidth; ++x)
        {
            // Cancel processing if requested by user
            if ( cancel_ ) {
                return;
            }

            if (job.pixels.size() >= 64) {
                queueJob(job);
                job.pixels.clear();
            }

            Point point = {x,y};
            job.pixels.push_back(point);
        }
    }
    if (!job.pixels.empty())
        queueJob(job);

    updateTimer_.start();
}


void InteractiveMCPTPlugin::canceledJob(QString /*_jobId*/ ) {
	emit log(LOGERR, "Cancel Button");
	cancel_ = true;
}

void InteractiveMCPTPlugin::threadFinished() {

	// Stop the update timer
    //updateTimer_.stop();
    //updateTimer_.disconnect();

    // Last update of image
	updateImageWidget();

}

void InteractiveMCPTPlugin::updateImageWidget() {

    const Vec3d markerColors[] = { {0.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 0.0, 0.0} };

    // Generate Image from accumulated buffer and sample counter
    for (int y = 0; y < image_.height(); ++y)
    {
        for (int x = 0; x < image_.width(); ++x)
        {
            int index = x + image_.width() * y;
            Vec3d color = mAccumulatedColor[index];
            color /= mSamples[index];

            color.maximize(Vec3d(0.0, 0.0, 0.0));
            color.minimize(Vec3d(1.0, 1.0, 1.0));

            uint8_t left = mQueuedSamples[index];
            if (left > 0 && left <= sizeof(markerColors))
            {
                color = 0.5 * color + 0.5 * markerColors[left - 1];
            }

            image_.setPixel(QPoint(x,y), QColor::fromRgbF(color[0], color[1], color[2]).rgb());
        }
    }

    // update the widget
    imageLabel_->setPixmap(QPixmap::fromImage(image_));
    imageLabel_->resize(image_.size());
    imageWindow->adjustSize();

    // update Job queue

    for (std::vector<QFuture<void> >::iterator it = mRunningFutures.begin();
         it != mRunningFutures.end();)
    {
        if (it->isFinished()) {
            it = mRunningFutures.erase(it);
        } else {
            ++it;
        }
    }
    if (mRunningFutures.empty()) updateTimer_.stop();
}

InteractiveMCPTPlugin::Intersection InteractiveMCPTPlugin::intersectScene(const Ray& _ray)
{
    Intersection result;
    result.depth = FLT_MAX;
    for(PluginFunctions::ObjectIterator o_It( PluginFunctions::ALL_OBJECTS, DataType( DATA_TRIANGLE_MESH | DATA_SPHERE) ) ; o_It != PluginFunctions::objectsEnd(); ++o_It) {
        Intersection next;
        if (intersect(*( *o_It), _ray, next.position, next.normal, next.depth))
        {
            if (next.depth < result.depth)
            {
                result = next;
                result.material = o_It->materialNode()->material();
            }
        }
    }
    return result;
}


Color InteractiveMCPTPlugin::trace(const Ray& _ray, unsigned int _recursions) {

    unsigned int max_depth = 3;
    Color black(0.0f,0.0f,0.0f,1.0f);

    if (_recursions > max_depth)
        return black;

    Intersection hit = intersectScene(_ray);
    Ray mirrored = reflect(_ray, hit.position, hit.normal);

    if (hit.depth == FLT_MAX) return black;
    if( (hit.normal | (-_ray.direction)) < 0.0 ) return black;

    // Reflectance used for emittance
    Color emitted = float(hit.material.reflectance()) * Color(1.0f, 1.0f, 1.0f, 0.0f);

    // Russian Roulette, wether to use diffuse or glossy samples
    double diffuseReflectance = Sampling::brightness(hit.material.diffuseColor());
    double specularReflectance = Sampling::brightness(hit.material.specularColor());
    double totalReflectance = diffuseReflectance + specularReflectance;

    ACG:Vec3d sample;
    ((Sampling::random() * totalReflectance) <= diffuseReflectance)
        ? sample = Sampling::randomDirectionsCosTheta(1, hit.normal).front()
        : sample = Sampling::randomDirectionCosPowerTheta(1, mirrored.direction, hit.material.shininess()).front();
    double density = (diffuseReflectance / totalReflectance) * Sampling::densityCosTheta(hit.normal, sample)
                  + (specularReflectance / totalReflectance) * Sampling::densityCosPowerTheta(mirrored.direction, hit.material.shininess(), sample);

    Ray reflectedRay;
    reflectedRay.origin = hit.position;
    reflectedRay.direction = sample;
    double costheta = sample | hit.normal;

    Color reflected = BRDF::phongBRDF(hit.material, _ray.direction, reflectedRay.direction, hit.normal)
                      * trace(reflectedRay, _recursions + 1) * costheta / density;
    return (emitted + reflected);
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
    _t = ((_face.p0 - _ray.origin) | _normal) / (_normal | _ray.direction);
    if (_t <= EPS) return false;

    _intersection = _ray.origin + _t * _ray.direction;

    Vec3d bary;
    if (bary_coord(_intersection, _face.p0, _face.p1, _face.p2, bary))
        if (bary[0] > 0. && bary[1] > 0. && bary[2] > 0.)
            return true;

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
    if (mAccumulatedColor) delete[] mAccumulatedColor;
    Vec3d zeroVec(0.,0.,0.);
    mAccumulatedColor = new Vec3d[imageWidth * imageHeight];
    for (int y = 0; y < imageHeight; ++y)
        for (int x = 0; x < imageWidth; ++x)
            mAccumulatedColor[x + y * imageWidth] = zeroVec;

    if (mSamples) delete[] mSamples;
    mSamples = new uint32_t[imageWidth * imageHeight];
    memset(mSamples, 0, imageWidth * imageHeight * sizeof(uint32_t));

    if (mQueuedSamples) delete[] mQueuedSamples;
    mQueuedSamples = new uint8_t[imageWidth * imageHeight];
    memset(mQueuedSamples, 0, imageWidth * imageHeight * sizeof(uint8_t));

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
  Q_EXPORT_PLUGIN2( interactiveMCPTplugin , InteractiveMCPTPlugin );
#endif
