#include <iostream>
#include <stdio.h>
#include <highgui.h>
#include <cv.h>
#include <tesseract/baseapi.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;
using namespace tesseract;

void frameDeteccaoPlaca( IplImage* frame );

IplImage* encontraPlaca(CvSeq* contours, IplImage* frame_gray, IplImage* frame );

void trataImagemHLS(IplImage* frame);

void tesseractOCR(IplImage *placa, int x, int y, int width, int height);

void saveImg(IplImage *img);

double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 );

IplImage* sobelImg(IplImage* frame);

void showImage(IplImage* frame, const char* str);


int main(int argc, char** argv)
{

  IplImage *frame;
  clock_t startT, endT;
   CvCapture *capture = NULL;
  
  capture = cvCreateFileCapture( argv[1] );

  //capture = cvCaptureFromCAM( -1 );

  //frame = cvLoadImage(argv[1]);

  if(capture)
  {
    while(true)
    {
      frame = cvQueryFrame( capture );

      if( !frame )
      {
        // cout << "Imagem não foi encontrada" << endl;
        // return -1;
        break;   
      }
      // else
      // {
      //   showImage(frame, "Imagem original");

        if( !frame->imageSize==0 )
        {
            startT=clock();

            frameDeteccaoPlaca( frame );

            endT=clock();

            cout << (double) (endT-startT) / CLOCKS_PER_SEC << endl;

        }
        else
        { 

          cout << "Problema na imagem! Sem captura" << endl; 
        }

      // }
    }
  }
  cvReleaseCapture( &capture );

  cvDestroyWindow("Deteccao da placa");

  return 0;

}


void frameDeteccaoPlaca( IplImage* frame )
{

  //trata imagem utilizando o tipo HLS (HUE, Light, Saturation)
  //trataImagemHLS(frame);
  //IplImage* frame_gray_s = cvCreateImage( cvSize( frame->width,frame->height ), IPL_DEPTH_8U, 1); // gray smooth

  IplImage* frame_gray = cvCreateImage( cvSize( frame->width,frame->height ), IPL_DEPTH_8U, 1);

  IplImage* frame_canny = cvCreateImage( cvSize( frame->width,frame->height ), IPL_DEPTH_8U, 1);

  IplImage* frame_canny2 = cvCreateImage( cvSize( frame->width,frame->height ), IPL_DEPTH_8U, 1);

  IplImage* frame_merge = cvCreateImage( cvSize( frame->width,frame->height ), IPL_DEPTH_8U, 1);

  // Escala cinza
  cvCvtColor( frame, frame_gray, CV_BGR2GRAY );
  //showImage(frame_gray, "Escala de cinza");


  // Retira ruidos
  cvSmooth(frame_gray, frame_gray, CV_GAUSSIAN, 3,3);
  //showImage(frame_gray, "Retirada de resíduos");

  //Sobel
  //IplImage* frame_sobel = cvCreateImage( cvSize( frame->width,frame->height ), IPL_DEPTH_8U, 1);

  //frame_sobel = sobelImg(frame_gray);

  //cvConvertScaleAbs(frame_sobel, frame_sobel);

  //cvShowImage("Sobel", frame_sobel);

  // Filtros canny cinza e canny colorida

  cvCanny( frame_gray, frame_canny, 0, 255, 3);
  
  //cvCanny( frame, frame_canny2, 0, 70, 3 );
  //showImage(frame_canny, "Canny");

  // Soma dos filtros

  //cvAddWeighted(frame_canny, 0.5, frame_canny2, 0.5, 0, frame_merge);

  // declara contorno e area na memoria

  CvSeq* contours = 0;

  CvMemStorage* storage = NULL;

  if( storage==NULL ) storage = cvCreateMemStorage(0);
  else cvClearMemStorage(storage);

  cvFindContours( frame_canny, storage, &contours, sizeof(CvContour),
      CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
  
  //cvSaveImage("canny1.jpeg", frame_canny);

  //-- Mostra imagem
  cvShowImage("Deteccao da placa", encontraPlaca(contours,frame_gray,frame) );
  waitKey(0);
  //cvShowImage( frame, "Deteccao da placa");

  // Libera recursos
  cvReleaseImage(&frame_gray);

  //cvReleaseImage(&frame_gray_s);

  cvReleaseImage(&frame_canny);

  cvReleaseImage(&frame_canny2);

  cvReleaseImage(&frame_merge);

  cvReleaseMemStorage(&storage);

}

IplImage* encontraPlaca(CvSeq* contours, IplImage* frame_gray,IplImage* frame )
{

  for ( ;contours!= NULL; contours = contours->h_next)
  {

      CvSeq* approxContour = cvApproxPoly(contours, contours->header_size , 
          contours->storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.05, 0);
      // cout << "===========================================" << endl;
      // cout << "approxContour->total = " << approxContour->total << endl;
      // cout << "cvContourArea " << fabs(cvContourArea(approxContour,CV_WHOLE_SEQ)) << endl;
      // cout << "cvCheckContourConvexity " << cvCheckContourConvexity(approxContour) << endl;
      // cout << "===========================================" << endl;


      if (approxContour->total >=4 && 
          fabs(cvContourArea(approxContour,CV_WHOLE_SEQ)) > 2000 &&
          cvCheckContourConvexity(approxContour))
      {
          CvBox2D box = cvMinAreaRect2(approxContour);

          double whRatio = (double)box.size.width / box.size.height;
          // cout << "whRatio = " << whRatio << endl;

          if ((!(2.7 < whRatio && whRatio < 3.4)) ) /*|| abs(box.angle)>20 razao da largura e altura do objeto (essa proporcao é 
                  para o formato brasileiro).*/
          {

              CvSeq* child = contours->v_next;

              if (child != NULL)
              encontraPlaca(child, frame_gray, frame); 

              continue;
          }

          // cout << "whRatio = " << whRatio << endl;

          double s = 0;

          for( int i = 0; i <= approxContour->total; i++ )
          {
              // find minimum angle between joint
              // edges (maximum of cosine)

              if( i >= 2 )
              {
                  double t = fabs(angle(
                      (CvPoint*)cvGetSeqElem(approxContour, i),
                      (CvPoint*)cvGetSeqElem(approxContour, i-2), 
                      (CvPoint*)cvGetSeqElem(approxContour, i-1))); 
                  // cout << "t = " << t << endl;

                  s = s > t ? s : t;
                  // cout << "s = " << s << endl;

              }

          }

          // if cosines of all angles are small
          // (all angles are ~90 degree) then write quandrange
          // vertices to resultant sequence

          if( !(s < 0.3) )
          {

              CvSeq* child = contours->v_next;

              if (child != NULL)
                  encontraPlaca(child, frame_gray, frame); 

              continue;

          }
          
          // desenha retangulo vermelho

          cvDrawContours( frame , approxContour, CV_RGB(255,0,0),
              CV_RGB(255,0,0), -1, CV_FILLED, 8 );

          //amplia imagem

          CvRect box2 = cvBoundingRect(approxContour);

          int m_width = (int)9000/box2.width; //multiplicador de largura

          int height_size, width_size, width_char, height_char;

          if (m_width < 1) 
          {

              height_size = 300;
              width_size = 900;
              height_char = 159; //53%
              width_char = 102; //11.4%
          }
          else
          {

              height_size = m_width * box2.height / 10;
              width_size = m_width * box2.width / 10;
              height_char = height_size * 46/100; // altura da sessao dos caracteres
              width_char = width_size * 12/100; // largura 

          }

          IplImage* licenca = NULL;

          licenca = cvCreateImage( cvSize( width_size, height_size ), IPL_DEPTH_8U, 1);
          
          //recorta imagem
          cvSetImageROI(frame_gray, box2);
          cvResize(frame_gray, licenca, CV_INTER_LINEAR);
          cvResetImageROI(frame_gray);

          //trata placa comum

          cvThreshold(licenca,licenca,0,255,CV_THRESH_TRUNC | CV_THRESH_OTSU);
          cvThreshold(licenca,licenca,0,255, CV_THRESH_BINARY | CV_THRESH_OTSU);
          
          //trata placa colorida

          //cvThreshold(licenca,licenca,0,255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

          cvDilate(licenca,licenca,NULL,4);

          cvErode(licenca,licenca,NULL,2);

          //separa os caracteres

          CvRect box_char;
          box_char.x = width_size*5/100;
          box_char.y = height_size*33/100;
          box_char.width = width_size*88/100;
          box_char.height = height_size*57/100;
          
          //saveImg(licenca);

          showImage(licenca, "Placa");
          // tesseractOCR(licenca, box_char.x, box_char.y ,
          //     box_char.width , box_char.height);
      }

      cvClearSeq(approxContour);

  }


  return frame;
}


void tesseractOCR(IplImage *placa, int x, int y, int width, int height)
{
  TessBaseAPI tess;

  tess.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPRSTUVWXYZ1234567890");

  tess.Init(NULL,"placa" );

  tess.SetImage((uchar*)placa->imageData, placa->width, placa->height,
      placa->nChannels, placa->widthStep);

  tess.SetRectangle(x,y,width,height);

  //tess.GetBoxText(0);

  tess.Recognize(0);

  //tess.TesseractRect

  const char* out = tess.GetUTF8Text();

  tess.End();
  
  cout << out << endl;

}

void trataImagemHLS(IplImage* frame)
{

  //transforma para o formato HLS

  IplImage* frame_HLS = cvCreateImage( cvSize( frame->width,
      frame->height ), IPL_DEPTH_8U, 3);

  cvCvtColor( frame, frame_HLS, CV_BGR2HLS );
  
  /**
   * Pega os valores da imagem de entrada
   */
  unsigned int height = frame_HLS->height;
  unsigned int width = frame_HLS->width;
  unsigned int step = frame_HLS->widthStep; 
  unsigned int channels= frame_HLS->nChannels;
  uchar *data = (uchar*) frame_HLS->imageData;
  /**
   * Aqui vamos varrer os pixels da imagem.
   * No for externo percorremos a altura e no
   * interno a largura.
   */

  for(unsigned int i=0; i < height; i++)
  {
      for(unsigned int j=0; j < width; j++){
          
          float H = float(data[i*step + j*channels]); // escala de 295 0 a 360
          float L = float(data[i*step + j*channels+1]); // escala 297 de 0 a 100
          float S = float(data[i*step + j*channels+2]); // escala 299 de 0 a 100
  
          // o branco fica mais cinza
          //if (!( (S <=30) && (L >= 70) )){
          // H = 0;
          // L = 0;
          // S = 0;
          //}

          // o verde fica branco
          //if ( !((H >= 0) && (H <=140) && (S >= 50)) ){
              // L = 100;
              // S = 0;
          //}

          // o azul fica branco
          //if ((H >= 90) && (H <=125) &&(L >= 0) && (S >= 100) ){
              // L = 255;
              // S = 0;
          //}

          // o vermelho fica branco
          //if ( (H>=330) && (H <=20) && (L >= 50) && (S >= 50)){
              // L = 100;
              // S = 0;
          //}

          data[i*step + j*channels]= uchar(H);
          data[i*step + j*channels+1]= uchar(L);

          data[i*step + j*channels+2]= uchar(S);
      }

  }

  //Converte para o formato BGR (padrao)
  cvCvtColor( frame_HLS, frame, CV_HLS2BGR );
  cvReleaseImage(&frame_HLS);

}

double angle( CvPoint* pt1, CvPoint* pt2, CvPoint* pt0 )
{
  double dx1 = pt1->x - pt0->x;
  double dy1 = pt1->y - pt0->y;
  double dx2 = pt2->x - pt0->x;
  double dy2 = pt2->y - pt0->y;
  
  return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}


void saveImg(IplImage *img)
{
  time_t t = time(0);
  struct tm tstruct;
  char buf[80];
  
  tstruct = *localtime (&t);

  strftime(buf, sizeof(buf), "%m-%d-%H-%M-%S.jpg" , &tstruct);

  cvSaveImage(buf,img);

}

IplImage* sobelImg(IplImage* frame) //utilizando a escala de cinza
{
  IplImage* frame_x = cvCreateImage( cvSize( frame->width,
      frame->height ), IPL_DEPTH_8U, 1);

  IplImage* frame_y = cvCreateImage( cvSize( frame->width,
      frame->height ), IPL_DEPTH_8U, 1);

  IplImage* frame_sobel = cvCreateImage( cvSize( frame->width,
      frame->height ), IPL_DEPTH_8U, 1);
  
  //em X
  cvSobel(frame, frame_x, 2, 0);
  
  //em Y
  cvSobel(frame, frame_y, 0, 2);
  
  cvAddWeighted( frame_x, 0.5, frame_y, 0.5, 0, frame_sobel );
  
  cvReleaseImage(&frame_x);

  cvReleaseImage(&frame_y);

  return frame_sobel;
}

void showImage(IplImage* frame, const char* str)
{

  namedWindow( str, WINDOW_AUTOSIZE );// Create a window for display.
  cvShowImage( str, frame );                   // Show our image inside it.

  waitKey(0);
  cvDestroyWindow(str);

}

