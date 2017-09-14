//
//  ofxKinectV2.cpp
//  kinectExample
//
//  Created by Theodore Watson on 6/23/14.
//
//

#include "ofxKinectV2.h"

//--------------------------------------------------------------------------------
ofxKinectV2::ofxKinectV2(){
    bNewFrame  = false;
    bNewBuffer = false;
    bOpened    = false;
    bUseRegistration = false;
    bMapDepthPixels = true;
    lastFrameNo = -1;
    
    //set default distance range to 50cm - 600cm
    
    params.add(minDistance.set("minDistance", 500, 0, 12000));
    params.add(maxDistance.set("maxDistance", 6000, 0, 12000));
}

//--------------------------------------------------------------------------------
ofxKinectV2::~ofxKinectV2(){
    close();
}

//--------------------------------------------------------------------------------
static bool sortBySerialName( const ofxKinectV2::KinectDeviceInfo & A, const ofxKinectV2::KinectDeviceInfo & B ){
    return A.serial < B.serial;
}

//--------------------------------------------------------------------------------
vector <ofxKinectV2::KinectDeviceInfo> ofxKinectV2::getDeviceList(){
    vector <KinectDeviceInfo> devices;
    
    int num = protonect.getFreenect2Instance().enumerateDevices();
    for (int i = 0; i < num; i++){
        KinectDeviceInfo kdi;
        kdi.serial = protonect.getFreenect2Instance().getDeviceSerialNumber(i);
        kdi.freenectId = i; 
        devices.push_back(kdi);
    }
    
    ofSort(devices, sortBySerialName);
    for (int i = 0; i < num; i++){
        devices[i].deviceId = i;
    }
    
    return devices;
}

//--------------------------------------------------------------------------------
unsigned int ofxKinectV2::getNumDevices(){
   return getDeviceList().size(); 
}

//--------------------------------------------------------------------------------
bool ofxKinectV2::open(unsigned int deviceId){
    
    vector <KinectDeviceInfo> devices = getDeviceList();
    
    if( devices.size() == 0 ){
        ofLogError("ofxKinectV2::open") << "no devices connected!";
        return false;
    }
    
    if( deviceId >= devices.size() ){
        ofLogError("ofxKinectV2::open") << " deviceId " << deviceId << " is bigger or equal to the number of connected devices " << devices.size() << endl;
        return false;
    }

    string serial = devices[deviceId].serial;
    return open(serial);
}

//--------------------------------------------------------------------------------
bool ofxKinectV2::open(string serial){
    close(); 
    
    params.setName("kinectV2 " + serial);
    
    bNewFrame  = false;
    bNewBuffer = false;
    bOpened    = false;
    
    int retVal = protonect.openKinect(serial);
    
    if(retVal==0){
        lastFrameNo = -1;
        startThread(true);
    }else{
        return false;
    }
    
    bOpened = true;
    return true;
}

//--------------------------------------------------------------------------------
void ofxKinectV2::threadedFunction(){

  while(isThreadRunning()){

    if (bUseRegistration) {
      protonect.updateKinect(rgbPixelsBack, depthPixelsBack, &depthPixelsUndistortedBack, NULL);
      //protonect.updateKinect(rgbPixelsBack, depthPixelsBack, &depthPixelsUndistortedBack, &depthOnRgbPixelsBack);
    }
    else {
      protonect.updateKinect(rgbPixelsBack, depthPixelsBack);
    }

    rgbPixelsFront.swap(rgbPixelsBack);
    depthPixelsFront.swap(depthPixelsBack);

    if (bUseRegistration) {
      depthPixelsUndistortedFront.swap(depthPixelsUndistortedBack);
      //depthOnRgbPixelsFront.swap(depthOnRgbPixelsBack);
    }

    lock();
    bNewBuffer = true;
    unlock();
  }
}

//--------------------------------------------------------------------------------
void ofxKinectV2::update(){
    if( ofGetFrameNum() != lastFrameNo ){
        bNewFrame = false;
        lastFrameNo = ofGetFrameNum();
    }
    if( bNewBuffer ){
    
        lock();
            rgbPix = rgbPixelsFront;
            rawDepthPixels = depthPixelsFront;

            if (bUseRegistration) {
              rawDepthPixelsUndistorted = depthPixelsUndistortedFront;
              //rawDepthOnRgbPixels = depthOnRgbPixelsFront;
            }

            bNewBuffer = false;
        unlock();

        if (bMapDepthPixels)
          mapDepthPixels(rawDepthPixels, depthPix);

        if (bUseRegistration && bMapDepthPixels) {
          mapDepthPixels(rawDepthPixelsUndistorted, depthPixUndistorted);
          //mapDepthPixels(rawDepthOnRgbPixels, depthOnRgbPixels);
        }

        bNewFrame = true; 
    }
}

void ofxKinectV2::mapDepthPixels(const ofFloatPixels& raw, ofPixels& mapped) {

  if ( raw.size() <= 0 ) 
    return;

  if ( mapped.getWidth() != raw.getWidth() )
    mapped.allocate(raw.getWidth(), raw.getHeight(), 1);

  //const float * pixelsF = raw.getData();
  unsigned char * pixels = mapped.getData();

  for (int i = 0; i < mapped.size(); i++) {
    pixels[i] = ofMap(raw[i], minDistance, maxDistance, 255, 0, true);
    if ( pixels[i] == 255 )
      pixels[i] = 0;
  }
};

//--------------------------------------------------------------------------------
bool ofxKinectV2::isFrameNew(){
    return bNewFrame; 
}

//--------------------------------------------------------------------------------
ofPixels ofxKinectV2::getDepthPixels(){
    return depthPix;
}

//--------------------------------------------------------------------------------
ofFloatPixels ofxKinectV2::getRawDepthPixels(){
    return rawDepthPixels;
}

//--------------------------------------------------------------------------------

ofFloatPixels ofxKinectV2::getRawDepthPixelsUndistorted(){
    return rawDepthPixelsUndistorted;
}

ofPixels ofxKinectV2::getDepthPixelsUndistorted(){
    return depthPixUndistorted;
}

//ofFloatPixels ofxKinectV2::getRawDepthOnRgbPixels(){
    //return rawDepthOnRgbPixels;
//}

//ofPixels ofxKinectV2::getDepthOnRgbPixels(){
    //return depthOnRgbPixels;
//}

//--------------------------------------------------------------------------------
ofPixels ofxKinectV2::getRgbPixels(){
    return rgbPix; 
}

//--------------------------------------------------------------------------------

ofxKinectV2::PointXYZRGB ofxKinectV2::getPointXYZRGB(int x, int y) const {
  float _x, _y, _z;
  float rgb;
  protonect.getPointXYZRGB(x, y, _x, _y, _z, rgb);
  const uint8_t *p = reinterpret_cast<uint8_t*>(&rgb);
  uint8_t b = p[0];
  uint8_t g = p[1];
  uint8_t r = p[2];
	ofxKinectV2::PointXYZRGB point;
  point.xyz = ofVec3f(_x, _y, _z);
  point.rgb = ofColor(r, g, b);
  return point;
}

ofVec3f ofxKinectV2::getWorldCoordinateAt(int x, int y) const {
  //float _x, _y, _z;
  //protonect.getPointXYZ(x, y, _x, _y, _z);
  //return ofVec3f(_x, _y, _z);
  ofxKinectV2::PointXYZRGB point = getPointXYZRGB(x, y);
  return point.xyz;
}

ofColor ofxKinectV2::getColorAt(int x, int y) const {
  ofxKinectV2::PointXYZRGB point = getPointXYZRGB(x, y);
  return point.rgb;
}

void ofxKinectV2::setRegistration(bool bUseRegistration){
  this->bUseRegistration = bUseRegistration;
}

void ofxKinectV2::setMapDepthPixels(bool bMapDepthPixels){
  this->bMapDepthPixels = bMapDepthPixels;
}

//--------------------------------------------------------------------------------
void ofxKinectV2::close(){
    if( bOpened ){
        waitForThread(true);
        protonect.closeKinect();
        bOpened = false;
    }
}


