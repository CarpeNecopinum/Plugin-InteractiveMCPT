#ifndef INTERACTIVEMCPT_HH
#define INTERACTIVEMCPT_HH

#include <OpenFlipper/BasePlugin/BaseInterface.hh>
#include <OpenFlipper/BasePlugin/LoggingInterface.hh>
#include <OpenFlipper/BasePlugin/ToolboxInterface.hh>
#include <OpenFlipper/BasePlugin/ProcessInterface.hh>
#include <OpenFlipper/common/Types.hh>
#include <QRunnable>

#include <QTimer>
#include <stdint.h>

#include <ObjectTypes/TriangleMesh/TriangleMesh.hh>
#include <ObjectTypes/Light/Light.hh>

#include "MonteCuda.hh"
#include "InfoStructs.hh"

class ImageViewer;

typedef ACG::Vec3d Vec3d;


struct Face
{
  Face(TriMesh& _mesh, TriMesh::FaceHandle _fh ) {
    TriMesh::FaceVertexIter fv_it(_mesh, _fh);
    p0 = _mesh.point(*fv_it);
    ++fv_it;
    p1 = _mesh.point(*fv_it);
    ++fv_it;
    p2 = _mesh.point(*fv_it);

  }

  Vec3d p0;
  Vec3d p1;
  Vec3d p2;

};

typedef ACG::SceneGraph::Material Material;
typedef ACG::Vec4f Color;


class InteractiveMCPTPlugin : public QObject, BaseInterface, LoggingInterface, ToolboxInterface, ProcessInterface
{
  Q_OBJECT
  Q_INTERFACES(BaseInterface)
  Q_INTERFACES(LoggingInterface)
  Q_INTERFACES(ToolboxInterface)
  Q_INTERFACES(ProcessInterface)
#if QT_VERSION >= 0x050000
  Q_PLUGIN_METADATA(IID "org.OpenFlipper.Plugins.Plugin-InteractiveMCPT")
#endif

  signals:
    //BaseInterface
    void updateView();

    //LoggingInterface
    void log(Logtype _type, QString _message);
	void log(QString _message);

    // ToolboxInterface
    void addToolbox(QString _name, QWidget* _toolbox, QIcon* icon);

    // ProcessInterface
    void startJob( QString _jobId, QString _description , int _min , int _max , bool _blocking);
    void setJobState( QString _jobId, int _value );
    void finishJob(QString _jobID );

  public:
    // BaseInterface
    QString name() { return (QString("Interactive MCPT")); }
    QString description( ) { return (QString("")); }

private slots:

    void initializePlugin(); // BaseInterface
    
	void openWindow();
    void threadFinished();
    void canceledJob(QString /*_jobId*/ );
    void updateImageWidget();
    void showContextMenu(QPoint _point);
    void saveImage();

    void globalRender();

	void selectBrushBtnPressed();
    void changeRaysPerPixel(int rays) { mSettings.samplesPerPixel = rays; }
	void changeBrushSize(int size);
	void changeBrushDepth(int depth);

    bool intersectBoundingBox(const Vec3d& bb_min ,
                              const Vec3d& bb_max ,
                              const Ray& _ray);
    bool intersect(BaseObjectData&      _object,
                   const Ray&           _ray,
                   Vec3d&               _intersection,
                   Vec3d&               _normal,
                   double&              _t );

   /** \brief compute color from ray
   *
   * This function shoots a ray into the scene and computes the visible color along this ray.
   * The _depth parameter restricts the number of recursions for mirrored rays.
   * See source file for implementation details
   */
   Color trace(const Ray& _ray, unsigned int _recursions);

   /** \brief Compute a reflection ray
   *
   * This function reflects the given ray at the point _point with the normal _normal.
   * See source file for implementation details
   */
   Ray reflect(const Ray&   _ray,
               const Vec3d& _point,
               const Vec3d& _normal);

   /** \brief Intersect a ray with a Triangle
   *
   * This function intersects a ray with a given face.
   * It returns true if the face has been intersected. Additionally it returns the intersection point,normal and
   * the intersection parameter.
   * See source file for implementation details
   */
   bool intersectTriangle( const Face&        _face,
                           const Vec3d&       _normal,
                           const Ray&         _ray,
                           Vec3d&             _intersection,
                           double&            _t ) const;

   /** \brief Intersect a ray with a Sphere
   *
   * This function intersects a ray with a given sphere.
   * It returns true if the sphere has been intersected. Additionally it returns the intersection point,normal and
   * the intersection parameter.
   * See source file for implementation details
   */
   bool intersectSphere( const Vec3d&      _center,
                         const double&     _radius,
                         const Ray&        _ray,
                         Vec3d&            _intersection,
                         Vec3d&            _normal,
                         double&           _t ) const;

   /** \brief Compute barycentric coordinates
   *
   * This function computes the barycentric coordinates of the point p in the triangle
   * u,v,w and returns the result in _result.
   */
   bool bary_coord( const Vec3d &  _p,
                    const Vec3d &  _u,
                    const Vec3d &  _v,
                    const Vec3d &  _w,
                    Vec3d &        _result ) const;


   public slots:
      QString version() { return QString("1.0"); }
      void testMousePressed(QMouseEvent* ev);
	  void testMouseReleased(QMouseEvent* ev);

	  void testFocusIn(QEvent* ev);
	  void testFocusOut(QEvent* ev);

	  void testMouseMove(QMouseEvent* ev);

public:

	ImageViewer* getImageViewer(){
		return imageLabel_;
	}

	QTimer &getUpdateTimer(){
		return updateTimer_;
	}

	struct RenderSettings
	{
		int samplesPerPixel;
	};

	struct RenderJob
	{
		RenderSettings settings;
		std::vector<Point> pixels;
	};

	void queueJob(RenderJob job);

protected:
      CameraInfo mCam;

      CameraInfo computeCameraInfo() const;
      Vec3d* mAccumulatedColor;
      uint32_t* mSamples;
      uint8_t* mQueuedSamples;

      RenderSettings mSettings;


      struct Intersection
      {
          Vec3d position;
          Vec3d normal;
          double depth;
          Material material;
      };
      Intersection intersectScene(const Ray &_ray);
      void runJob(RenderJob job);
      std::vector<QFuture<void> > mRunningFutures;

private:

	QTimer updateTimer_;
     // The rendered image
     QImage image_;
     ImageViewer* imageLabel_;
     QWidget* imageWindow;


     // Light sources in the scene
     std::vector< LightSource > lights_;

     // If processing has to be canceled, this variable is set to true
     bool cancel_;

     void clearImage();

     void tracePixel(size_t x, size_t y);

	 void initializeDrawingGUI(QGridLayout* layout, QWidget* parent = 0); // Called in initializePlugin.
};

#endif //INTERACTIVEMCPT_HH
