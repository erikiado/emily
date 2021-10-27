   #include "Frame.hpp"
   #include "Segmentation.hpp"
   #include "Trail.hpp"
   
   static void usage(const char *name)
   {
           std::cout << "usage: " << name << " [OPTIONS]" << std::endl;
           std::cout << "OPTIONS:" << std::endl;
           std::cout << "\t-i <file>\tthe input file (if missing, a webcam will be tried)" << std::endl;
           std::cout << "\t-o <file>\tthe optional output file" << std::endl;
           std::cout << "\t-l <number>\tthe trail length in frames" << std::endl;
           std::cout << "\t\t\tthe default is 25" << std::endl;
           std::cout << "\t\t\tNOTES:" << std::endl;
           std::cout << "\t\t\t  * a negative value means an 'infinite' trail" << std::endl;
           std::cout << "\t-s <method>\tthe image segmentation method" << std::endl;
           std::cout << "\t\t\tvalid values are:" << std::endl;
           std::cout << "\t\t\t none, threshold, background" << std::endl;
           std::cout << "\t\t\tthe default is 'background'" << std::endl;
           std::cout << "\t\t\tNOTES:" << std::endl;
           std::cout << "\t\t\t  * 'none' is only useful with '-d average'" << std::endl;
           std::cout << "\t-b <number>\tthe number of initial frames for background learning," << std::endl;
           std::cout << "\t\t\tthe default is 50" << std::endl;
           std::cout << "\t\t\tNOTES:" << std::endl;
           std::cout << "\t\t\t  * only useful with '-s background'" << std::endl;
           std::cout << "\t-t <number>\tthe level for the threshold segmentation method," << std::endl;
           std::cout << "\t\t\tthe default is 5" << std::endl;
           std::cout << "\t\t\tNOTES:" << std::endl;
           std::cout << "\t\t\t  * only useful with '-s threshold'" << std::endl;
           std::cout << "\t-d <method>\tthe trail drawing method" << std::endl;
           std::cout << "\t\t\tvalid values are:" << std::endl;
           std::cout << "\t\t\t  copy, accumulate, fadecopy, fadeaccumulate, average" << std::endl;
           std::cout << "\t\t\tthe default is 'copy'" << std::endl;
           std::cout << "\t\t\tNOTES:" << std::endl;
           std::cout << "\t\t\t  * 'copy' is useless with '-s none'" << std::endl;
           std::cout << "\t\t\t  * the difference between 'fadecopy' and" << std::endl;
           std::cout << "\t\t\t    'fadeaccumulate is cleared when using '-B'" << std::endl;
           std::cout << "\t-r\t\treverse the trail drawing sequence" << std::endl;
           std::cout << "\t-B\t\tshow the background behind the trail" << std::endl;
           std::cout << "\t\t\tNOTES:" << std::endl;
           std::cout << "\t\t\t  * only used with '-s background'" << std::endl;
           std::cout << "\t-F\t\tredraw the current frame on top of the trail" << std::endl;
           std::cout << "\t\t\tNOTES:" << std::endl;
           std::cout << "\t\t\t  * noticeable with '-s average'" << std::endl;
           std::cout << "\t\t\t  * noticeable with reverse faded trails" << std::endl;
   }
   
   int main(int argc, char *argv[])
   {
           int ret = 0;
           int opt;
   
           std::string *input_file = NULL;
           std::string *output_file = NULL;
           int trail_lenght = 25;
           std::string *segmentation_method = new std::string("background");
           int background_learn_frames = 50;
           int threshold_level = 5;
           std::string *drawing_method = new std::string("copy");
           bool reverse_trail = false;
           bool show_background = false;
           bool redraw_current_frame = false;
   
           while ((opt = getopt(argc, argv, "i:o:l:s:b:t:d:rBFh")) != -1) {
                   switch (opt) {
                   case 'i':
                           input_file = new std::string(optarg);
                           break;
                   case 'o':
                           output_file = new std::string(optarg);
                           break;
                   case 'l':
                           trail_lenght = atoi(optarg);
                           break;
                   case 's':
                           delete segmentation_method;
                           segmentation_method = new std::string(optarg);
                           break;
                  case 'b':
                          background_learn_frames = atoi(optarg);
                          break;
                  case 't':
                          threshold_level = atoi(optarg);
                          break;
                  case 'd':
                          delete drawing_method;
                          drawing_method = new std::string(optarg);
                          break;
                  case 'r':
                          reverse_trail = true;
                          break;
                  case 'B':
                          show_background = true;
                          break;
                  case 'F':
                          redraw_current_frame = true;
                          break;
                  case 'h':
                          usage(argv[0]);
                          return 0;
                  default: /* '?' */
                          usage(argv[0]);
                          return -1;
                  }
          }
  
          cv::VideoCapture inputVideo;
          cv::VideoWriter outputVideo;
          cv::Size frame_size;
          cv::Mat input_frame;
  
          if (input_file) {
                  inputVideo.open(*input_file);
          } else {
                  inputVideo.open(0);
          }
  
          if (!inputVideo.isOpened()) {
                  std::cerr  << "Could not open the input video." << std::endl;
                  ret = -1;
                  goto out;
          }
  
          frame_size = cv::Size((int) inputVideo.get(CV_CAP_PROP_FRAME_WIDTH),
                                (int) inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
  
          if (output_file) {
                  int fps = inputVideo.get(CV_CAP_PROP_FPS);
                  if (fps < 0)
                          fps = 25;
  
                  outputVideo.open(*output_file, CV_FOURCC('M','J','P','G'), fps, frame_size, true);
                  if (!outputVideo.isOpened()) {
                          std::cerr  << "Could not open the output video for write." << std::endl;
                          ret = -1;
                          goto out;
                  }
          }
  
          Trail *trail;
          if (reverse_trail)
                  trail = new ReverseTrail(trail_lenght, frame_size);
          else
                  trail = new ForwardTrail(trail_lenght, frame_size);
  
          trail->setRedrawCurrentFrame(redraw_current_frame);
  
          if (trail->setDrawingMethod(*drawing_method) < 0) {
                  std::cerr  << "Invalid drawing method." << std::endl;
                  ret = -1;
                  goto out_delete_trail;
          }
  
          Segmentation *segmentation;
          if (*segmentation_method == "background") {
                  segmentation = new MOG2Segmentation(inputVideo, background_learn_frames);
                  if (show_background) {
                          cv::Mat background(frame_size, inputVideo.get(CV_CAP_PROP_FORMAT));
  
                          ((MOG2Segmentation *)segmentation)->getBackgroundImage(background);
                          trail->setBackground(background);
                  }
          } else if (*segmentation_method == "threshold") {
                  segmentation = new ThresholdSegmentation(threshold_level);
          } else if (*segmentation_method == "none") {
                  segmentation = new DummySegmentation();
          } else {
                  std::cerr  << "Invalid segmentation method." << std::endl;
                  ret = -1;
                  goto out_delete_trail;
          }
  
          cv::namedWindow("Frame", CV_WINDOW_NORMAL);
  
          for (;;) {
                  inputVideo >> input_frame;
  
                  Frame *foreground = new Frame(input_frame,
                                                segmentation->getForegroundMask(input_frame));
                  trail->update(foreground);
  
                  cv::Mat canvas = cv::Mat::zeros(input_frame.size(), input_frame.type());
                  trail->draw(canvas);
  
                  cv::imshow("Frame", canvas);
                  if (cv::waitKeyEx(30) >= 0)
                          break;
  
                  if (outputVideo.isOpened())
                          outputVideo << canvas;
          }
  
          cv::destroyWindow("Frame");
  
          delete segmentation;
  
  out_delete_trail:
          delete trail;
  out:
          delete drawing_method;
          delete segmentation_method;
          delete output_file;
          delete input_file;
          return ret;
  }