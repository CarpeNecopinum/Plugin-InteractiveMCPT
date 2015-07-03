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
#include <QSlider>

#include "ImageViewer.hh"

#include "InteractiveDrawing.hh"

#define EPS (1e-6)

InteractiveDrawing mInteractiveDrawing;
QDoubleSpinBox * mSeSigma;
QLabel * mSigmaLabel;

Vec3d vecPow(const Vec3d& in, double exponent)
{
    return Vec3d(
                std::pow(in[0], exponent),
                std::pow(in[1], exponent),
                std::pow(in[2], exponent)
            );
}

void InteractiveMCPTPlugin::changeBrushType(int type){
    mInteractiveDrawing.switchBrush(type);
    if(type != 3){
        mSeSigma->setVisible(false);
        mSigmaLabel->setVisible(false);
    }else{
        mSeSigma->setVisible(true);
        mSigmaLabel->setVisible(true);
    }
}

void InteractiveMCPTPlugin::changeBrushSize(int size){
    mInteractiveDrawing.getBrush().setSize(size);
    mInteractiveDrawing.updateSigma();
}

void InteractiveMCPTPlugin::changeSigma(double sigma){
    mInteractiveDrawing.setSigma(sigma);
    mInteractiveDrawing.updateSigma();
}

void InteractiveMCPTPlugin::changeMaxAngleDev(double maxAngleDev){
    mSmoother.setMaxAngleDeviation(maxAngleDev);
}

void InteractiveMCPTPlugin::changeMaxDepthDev(double maxDepthDev){
    mSmoother.setMaxDepthDeviation(maxDepthDev);
}

void InteractiveMCPTPlugin::changeSmoothSigma(double smoothSigma){
    mSmoother.setSigma(smoothSigma);
}

void InteractiveMCPTPlugin::smooth(){
    mSmoother.smooth(this, mAccumulatedColor, mSamples);
}

void InteractiveMCPTPlugin::mousePressed(QMouseEvent *ev){
    mInteractiveDrawing.startBrushStroke();
}

void InteractiveMCPTPlugin::testMousePressed(QMouseEvent *ev){
	emit log(LOGERR, QString("MousePressed"));
    mInteractiveDrawing.traceBrush(this, ev->x(), ev->y());
}

void InteractiveMCPTPlugin::testFocusIn(QEvent* ev){
	emit log(LOGERR, QString("Focus In!"));
}

void InteractiveMCPTPlugin::testFocusOut(QEvent* ev){
	emit log(LOGERR, QString("Focus Out!"));
}

void InteractiveMCPTPlugin::mouseMove(QMouseEvent* ev){
    mInteractiveDrawing.updateBrushStroke(this, ev);
}

void InteractiveMCPTPlugin::initializeDrawingGUI(QGridLayout* layout, QWidget* parent){

    int currentRow = 0;

#ifdef HAS_CUDA
    // Cuda Checkbox
    QCheckBox* cudaCheckBox = new QCheckBox("Use Cuda for Rendering", parent);
    connect(cudaCheckBox, SIGNAL(stateChanged(int)), this, SLOT(setCudaActive(int)));
    layout->addWidget(cudaCheckBox, currentRow++, 0, 1, 2);
#else
    layout->addWidget(new QLabel("This would be more fun with Cuda!", parent), currentRow++, 0, 1, 2);
#endif

    // Global Render button
    QPushButton* globalRenderButton = new QPushButton("FullImage MCPT",parent);
    connect(globalRenderButton, SIGNAL(clicked()), this, SLOT(globalRender()));
    layout->addWidget(globalRenderButton, currentRow++, 0, 1, 2);

    // Smoothing button
    QPushButton* smoothingButton = new QPushButton("Smooth image",parent);
    connect(smoothingButton, SIGNAL(clicked()), this, SLOT(smooth()));
    layout->addWidget(smoothingButton, currentRow++, 0, 1, 2);

    //Smoothing Max Angle Devation
    QDoubleSpinBox* seMaxAngleDev = new QDoubleSpinBox(parent);
    seMaxAngleDev->setMaximum(1.57);
    seMaxAngleDev->setMinimum(0.0);
    seMaxAngleDev->setSingleStep(0.01);
    layout->addWidget(new QLabel("Maximum Angle \nDeviation in Rad", parent), currentRow, 0);
    layout->addWidget(seMaxAngleDev, currentRow++, 1);
    connect(seMaxAngleDev, SIGNAL(valueChanged(double)), this, SLOT(changeMaxAngleDev(double)));
    seMaxAngleDev->setValue(0.52);

    //Smoothing Max Depth Deviation
    QDoubleSpinBox* seMaxDepthDev = new QDoubleSpinBox(parent);
    seMaxDepthDev->setMaximum(1.0);
    seMaxDepthDev->setMinimum(0.0);
    seMaxDepthDev->setSingleStep(0.01);
    layout->addWidget(new QLabel("Relative Maximum\nDepth Deviation", parent), currentRow, 0);
    layout->addWidget(seMaxDepthDev, currentRow++, 1);
    connect(seMaxDepthDev, SIGNAL(valueChanged(double)), this, SLOT(changeMaxAngleDev(double)));
    seMaxDepthDev->setValue(0.1);

    //Smoothing Sigma
    QDoubleSpinBox* seSmoothSigma = new QDoubleSpinBox(parent);
    seSmoothSigma->setMaximum(100.0);
    seSmoothSigma->setMinimum(0.1);
    seSmoothSigma->setSingleStep(0.1);
    layout->addWidget(new QLabel("Sigma for \nGaussian Smoothing", parent), currentRow, 0);
    layout->addWidget(seSmoothSigma, currentRow++, 1);
    connect(seSmoothSigma, SIGNAL(valueChanged(double)), this, SLOT(changeSmoothSigma(double)));
    seSmoothSigma->setValue(10.00);

	//Brush GUI
    QComboBox* brushComboBox = new QComboBox(parent);
    brushComboBox->addItem(QString("None"));
    brushComboBox->addItem(QString("Square Brush"));
    brushComboBox->addItem(QString("Circle Brush"));
    brushComboBox->addItem(QString("Gaussed Cirlce Brush"));

    layout->addWidget(new QLabel("Selected Brush", parent), currentRow, 0);
    layout->addWidget(brushComboBox, currentRow++, 1);
    connect(brushComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(changeBrushType(int)));


    // Rays per Pixel spinbox
    QSpinBox * seRaysPerPixel = new QSpinBox(parent);
    seRaysPerPixel->setMaximum(4096);
    seRaysPerPixel->setMinimum(1);
    connect(seRaysPerPixel, SIGNAL(valueChanged(int)), this, SLOT(changeRaysPerPixel(int)));
    layout->addWidget(new QLabel("Rays per Pixel", parent), currentRow, 0);
    layout->addWidget(seRaysPerPixel, currentRow++, 1);

    // Brush Size
	QSpinBox * seBrushSize = new QSpinBox(parent);
    seBrushSize->setMaximum(200);
    seBrushSize->setMinimum(1);
    layout->addWidget(new QLabel("Brush Radius", parent), currentRow, 0);
    layout->addWidget(seBrushSize, currentRow++, 1);
	connect(seBrushSize, SIGNAL(valueChanged(int)), this, SLOT(changeBrushSize(int)));

    //Gauss Sigma
    mSeSigma = new QDoubleSpinBox(parent);
    mSeSigma->setMaximum(2.0);
    mSeSigma->setMinimum(0.2);
    mSeSigma->setSingleStep(0.01);
    mSigmaLabel = new QLabel("Gauss Sigma in Brush Sizes", parent);
    layout->addWidget(mSigmaLabel, currentRow, 0);
    layout->addWidget(mSeSigma, currentRow++, 1);
    connect(mSeSigma, SIGNAL(valueChanged(double)), this, SLOT(changeSigma(double)));
    mSeSigma->setValue(0.75);
    mSeSigma->setVisible(false);
    mSigmaLabel->setVisible(false);

    //"Tone Mapping" slider
    QSlider* toneSlider = new QSlider(Qt::Horizontal, parent);
    toneSlider->setMinimum(1); toneSlider->setMaximum(300);
    toneSlider->setValue(100);
    layout->addWidget(new QLabel("Brightness", parent), currentRow, 0);
    layout->addWidget(toneSlider, currentRow++, 1);
    connect(toneSlider, SIGNAL(valueChanged(int)), this, SLOT(changeTone(int)));

    // dummy stretch label
    layout->addWidget(new QLabel("", parent), currentRow, 0, 1, 2);
    layout->setRowStretch(4, 1);
}

void InteractiveMCPTPlugin::initializePlugin()
{
    mAccumulatedColor = 0;
    mSamples = 0;
    mQueuedSamples = 0;
	mSettings.samplesPerPixel = 1;

#ifdef HAS_CUDA
    std::cout << "Compiled with CUDA ... nice." << std::endl;
#else
    std::cout << "No CUDA here ... not nice." << std::endl;
#endif

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

    imageLabel_ = new ImageViewer(&image_, imageWindow);
	imageLabel_->setBackgroundRole(QPalette::Base);
	imageLabel_->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
	imageLabel_->setScaledContents(false);
	imageLabel_->setContextMenuPolicy(Qt::CustomContextMenu);
	
	connect(imageLabel_,SIGNAL(customContextMenuRequested(QPoint)),this,SLOT(showContextMenu(QPoint)));
    connect(imageLabel_,SIGNAL(mousePressed(QMouseEvent*)),this,SLOT(mousePresse(QMouseEvent*)));
    connect(imageLabel_, SIGNAL(mouseReleased(QMouseEvent*)), this, SLOT(mouseRelease(QMouseEvent*)));
    connect(imageLabel_, SIGNAL(mouseMoved(QMouseEvent*)), this, SLOT(mouseMove(QMouseEvent*)));
    connect(imageLabel_, SIGNAL(mouseEntered(QEvent*)), this, SLOT(testFocusIn(QEvent*)));
	connect(imageLabel_, SIGNAL(mouseLeaved(QEvent*)), this, SLOT(testFocusOut(QEvent*)));

	layout->addWidget(imageLabel_);

    QGridLayout * sideboxGrid = new QGridLayout(imageWindow);

	layout->addLayout(sideboxGrid);

	//gui and interactive stuff
	initializeDrawingGUI(sideboxGrid, imageWindow);

    connect(&updateTimer_,SIGNAL(timeout()),this,SLOT(updateImageWidget()) );
    updateTimer_.setInterval(333);
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

    PluginFunctions::ObjectIterator o_It( PluginFunctions::ALL_OBJECTS, DataType( DATA_TRIANGLE_MESH ));

#ifdef HAS_CUDA
    uploadGeometry(o_It, PluginFunctions::objectsEnd());
    uploadCameraInfo(mCam);
#endif

    mSmoother.init(this);

    imageWindow->show();
}

void InteractiveMCPTPlugin::cudaRunJob(RenderJob job)
{
#ifdef HAS_CUDA
    cudaTracePixels(job.pixels, mAccumulatedColor, mSamples, image_.width());

    std::vector<QueuedPixel>::iterator end = job.pixels.end();
    for (std::vector<QueuedPixel>::iterator it = job.pixels.begin(); it != end; ++it)
    {
        QueuedPixel& point = *it;
        size_t index = point.y * image_.width() + point.x;
        mQueuedSamples[index]--;
    }
    //updateImageWidget();
#else
    std::cerr << "You don't have compiled with CUDA you scrub!" << std::endl;
    std::exit(-1);
#endif
}

void InteractiveMCPTPlugin::runJob(RenderJob job)
{
    std::vector<QueuedPixel>::iterator end = job.pixels.end();
    for (std::vector<QueuedPixel>::iterator it = job.pixels.begin(); it != end; ++it)
    {
        QueuedPixel& queuedPixel = *it;

        for (int i = 0; i < queuedPixel.samples; i++)
            tracePixel(queuedPixel.x, queuedPixel.y);

        size_t index = queuedPixel.y * image_.width() + queuedPixel.x;
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

CameraInfo InteractiveMCPTPlugin::computeCameraInfo() const
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
    for (QueuedPixel point : job.pixels)
    {
        size_t index = point.y * image_.width() + point.x;
        mQueuedSamples[index]++;
    }

    if (mUseCuda)
    {
        size_t count = job.pixels.size();

        QueuedPixel p = { -1 , -1 , 0};
        while (job.pixels.size() % CUDA_BLOCK_SIZE != 0) job.pixels.push_back(p);

        mRunningFutures.push_back(QtConcurrent::run(this, &InteractiveMCPTPlugin::cudaRunJob, job));
        if (!updateTimer_.isActive()) updateTimer_.start();
    }
    else
    {
        mRunningFutures.push_back(QtConcurrent::run(this, &InteractiveMCPTPlugin::runJob, job));
        if (!updateTimer_.isActive()) updateTimer_.start();
    }
}


void InteractiveMCPTPlugin::globalRender()
{
    RenderJob job;
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

            if (job.pixels.size() >= 16384) {
                queueJob(job);
                job.pixels.clear();
            }

            QueuedPixel point = {x,y, mSettings.samplesPerPixel};
            job.pixels.push_back(point);
        }
    }
    if (!job.pixels.empty())
        queueJob(job);
}

void InteractiveMCPTPlugin::setCudaActive(int active) {
    mUseCuda = !!active;
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

	// InteractivDrawing Update:
    mInteractiveDrawing.update(this, imageLabel_);

    // Generate Image from accumulated buffer and sample counter
    for (int y = 0; y < image_.height(); ++y)
    {
        for (int x = 0; x < image_.width(); ++x)
        {
            int index = x + image_.width() * y;
            Vec3d color = mAccumulatedColor[index];
            color /= mSamples[index];

            color *= mTone;
            color.maximize(Vec3d(0.0, 0.0, 0.0));
            color.minimize(Vec3d(1.0, 1.0, 1.0));
            color = vecPow(color, 1.0 / 2.2);

            uint8_t left = mQueuedSamples[index];

            if (left > 0)
            {
                double alpha = ((double(left) / 10.0));
                color = (1.0 - alpha) * color + alpha * Vec3d(0.0, 1.0, 0.0);
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

    // Russian Roulette, whether to use diffuse or glossy samples
    double diffuseReflectance = Sampling::brightness(hit.material.diffuseColor());
    double specularReflectance = Sampling::brightness(hit.material.specularColor());
    double totalReflectance = diffuseReflectance + specularReflectance;

    Vec3d sample;
    double exponent = (double)hit.material.shininess() / 99.0 * 4096.0;
    ((Sampling::random() * totalReflectance) <= diffuseReflectance)
        ? sample = Sampling::randomDirectionsCosTheta(1, hit.normal).front()
        : sample = Sampling::randomDirectionCosPowerTheta(1, mirrored.direction, exponent).front();
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
