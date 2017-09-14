//
//  ofxKinectV2.h
//  kinectExample
//
//  Created by Theodore Watson on 6/23/14.
//
//

#pragma once

#include "ofProtonect.h"
#include "ofMain.h"

class ofxKinectV2 : public ofThread{

    public:
    
        struct KinectDeviceInfo{
            string serial;
            int deviceId;   //if you have the same devices plugged in device 0 will always be the same Kinect
            int freenectId; //don't use this one - this is the index given by freenect2 - but this can change based on order device is plugged in
        };

        struct PointXYZRGB
        {
          ofVec3f xyz;
          ofColor rgb;
        };

        ofxKinectV2();
        ~ofxKinectV2(); 
        
        //for some reason these can't be static - so you need to make a tmp object to query them
        vector <KinectDeviceInfo> getDeviceList();
        unsigned int getNumDevices();
    
        bool open(string serial);
        bool open(unsigned int deviceId = 0);
        void update();
        void close();
    
        bool isFrameNew();
    
        ofPixels getDepthPixels();
        ofPixels getRgbPixels();
        ofFloatPixels getRawDepthPixels();

        ofFloatPixels getRawDepthPixelsUndistorted();
        ofPixels getDepthPixelsUndistorted();

        //TODO depth on rgb mapping
        //registration->apply(..., bigdepth) takes too much cpu
        //https://github.com/OpenKinect/libfreenect2/issues/337
        //https://github.com/OpenKinect/libfreenect2/issues/509
        //ofFloatPixels getRawDepthOnRgbPixels();
        //ofPixels getDepthOnRgbPixels();

        PointXYZRGB getPointXYZRGB(int x, int y) const;
        ofVec3f getWorldCoordinateAt(int x, int y) const;
        ofColor getColorAt(int x, int y) const;
    
        ofParameterGroup params;
        ofParameter <float> minDistance;
        ofParameter <float> maxDistance;

        void setRegistration(bool bUseRegistration);

    protected:
        void threadedFunction();

        ofPixels rgbPix;
        ofPixels depthPix;
        ofFloatPixels rawDepthPixels;
    
        bool bNewBuffer;
        bool bNewFrame;
        bool bOpened;
        bool bUseRegistration;
    
        ofProtonect protonect; 
    
        ofPixels rgbPixelsBack;
        ofPixels rgbPixelsFront;
        ofFloatPixels depthPixelsBack;
        ofFloatPixels depthPixelsFront;

        ofFloatPixels rawDepthPixelsUndistorted;
        ofFloatPixels depthPixelsUndistortedBack;
        ofFloatPixels depthPixelsUndistortedFront;
        ofPixels depthPixUndistorted;

        //ofFloatPixels rawDepthOnRgbPixels;
        //ofFloatPixels depthOnRgbPixelsBack;
        //ofFloatPixels depthOnRgbPixelsFront;
        //ofPixels depthOnRgbPixels;

        int lastFrameNo; 

        void mapDepthPixels(const ofFloatPixels& raw, ofPixels& mapped);
};
