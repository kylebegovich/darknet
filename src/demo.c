#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define DEMO 1
#define SAVEVIDEO
// #define THREADLINE
#ifdef SAVEVIDEO
static CvVideoWriter *mVideoWriter;
#endif
#ifdef OPENCV

static char **demo_names;
static char **demo_names2;
static image **demo_alphabet;
static int demo_classes;
static int demo_classes2;

static float ***probs; // **probs to ***probs
static box **boxes; // *boxes to **boxes
static network net;
static network net2;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

//added counter
static int counter;
static info * result;

// added the number of instances for YOLO
static int YOLO = 2;

static int demo_frame = 5;
static int demo_detections = 0;
static int demo_detections2 = 0;
static float ***predictions; // **predictions to ***predictions
static int demo_index = 0;
static int demo_index2 = 0;
static int demo_done = 0;
static float *avg;
static float *avg2;
double demo_time;


double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void *detect_in_thread(void *ptr)
{
    running = 1;
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[0][demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions[0], demo_frame, l.outputs, avg);
    l.output = avg;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs[0], boxes[0], 0);
    } else if (l.type == REGION){
        get_region_boxes(l, buff[0].w, buff[0].h, net.w, net.h, demo_thresh, probs[0], boxes[0], 0, 0, 0, demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes[0], probs[0], l.w*l.h*l.n, l.classes, nms);

    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");
   
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}


void *detect_in_thread2(void *ptr)
{
    running = 1;
    float nms = .4;
    layer l = net2.layers[net2.n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    float *prediction = network_predict(net2, X);

    memcpy(predictions[1][demo_index2], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions[0], demo_frame, l.outputs, avg2);
    l.output = avg2;
    if(l.type == DETECTION){
         get_detection_boxes(l, 1, 1, demo_thresh, probs[1], boxes[1], 0);
    } else if (l.type == REGION){
         get_region_boxes(l, buff[1].w, buff[1].h, net2.w, net2.h, demo_thresh, probs[1], boxes[1], 0, 0, 0, demo_hier, 1);
    } else {
         error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes[1], probs[1], l.w*l.h*l.n, l.classes, nms);

    demo_index2 = (demo_index2 + 1)%demo_frame;
    running = 0;

    return 0;
}
void *fetch_in_thread(void *ptr)
{
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net.w, net.h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}


/* Function: draw_detection_thread
 * -------------------------
 * main function for drawing detection
 * 
 */
void* draw_detection_in_thread(void *ptr)
{
    image display = buff[(buff_index+2) % 3];
    /* duplicate function of draw_detections that writes the info of detected obj to result */
    draw_detections_info(display, demo_detections, demo_thresh, boxes[0], probs[0], 0, demo_names, demo_alphabet, demo_classes, result);
}

/*
 * Function:  counter_func
 * --------------------
 * a test function for multithread.
 * increments the counter and draw a vertical line respective to the counter
 * check if current counter is inbetween any detected box. If so, change color.
 *
 */
void *counter_func(void *ptr)
{
    counter += 1; /* increment the counter */

    image display = buff[(buff_index+2) % 3]; /* get the current frame image */

    /* set up color and width */
    float red = 0.1;
    float green = 0.5;
    float blue = 0.2;
    int width = 8;
    if(result[0].n > 0) /* check if any obj is detected */
    {
        printf("num detect %d\n", result[0].n);
        /* loop throug the object to check if x is bounded by x */
        for(int j = 1; j < result[0].n+1; j++){
            int x1 = result[j].left;
            int x2 = result[j].right;
            if(x1 <= counter && x2 >= counter){
                red = 1.0;
                break;
            }
        }
    }

    draw_vertical_line(display, counter, width, red, green, blue); /*draw vertical line respective to counter */

    if(counter >= display.w) counter = 0; /* reset counter if it went over width */
    return 0;
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    
    /*
    Prediction one set up 
    */
    image **alphabet = load_alphabet();    
    demo_frame = avg_frames;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_names = names;
    demo_thresh = thresh;
    demo_hier = hier;
    counter = 0; /*initilizating counter*/
    result = calloc(1000, sizeof(info)); /* initilizating info pointer assuming we will never detect more than 10000 items*/
    /* set this network to load on first gpu */
    cuda_set_device(0);
    net = parse_network_cfg(cfgfile);
    /* set the network default gpu to 0 */
    net.gpu_index = 0;
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    

    
    /*
    Prediction 2 set up 
    */
    if(YOLO == 2){
        char **names2 = get_labels("cfg/obj.names");
        char *cfgfile2 = "cfg/yolo-helmet-detect.cfg";
        char *weightfile2 = "yolo-helmet_10000.weights";
        demo_names2 = names2;
        demo_classes2 = 1;
        /* set this network to load on second gpu */
        cuda_set_device(1);        
        net2 = parse_network_cfg(cfgfile2);
        /* set the network default gpu to 1 */
        net2.gpu_index = 1;
        load_weights(&net2, weightfile2);
        set_batch_network(&net2, 1);
    }


    srand(2222222);
    
    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);

    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }
    /*
    * initilize the cv Video Writer
    */
    #ifdef SAVEVIDEO
    if(cap){
        int mfps = cvGetCaptureProperty(cap,CV_CAP_PROP_FPS);
        mVideoWriter = cvCreateVideoWriter("Output.avi",CV_FOURCC('M','J','P','G'),mfps,cvSize(cvGetCaptureProperty(cap,CV_CAP_PROP_FRAME_WIDTH),cvGetCaptureProperty(cap,CV_CAP_PROP_FRAME_HEIGHT)),1);
    }
    #endif

    if(!cap) error("Couldn't connect to webcam.\n");
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL);
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }
    


    layer l = net.layers[net.n-1];
    demo_detections = l.n*l.w*l.h;
    avg = (float *) calloc(l.outputs, sizeof(float));
    
    layer l2;
    if(YOLO==2){
        l2 = net2.layers[net2.n-1];
        demo_detections2 = l2.n*l2.w*l2.h;
        avg2 = (float *) calloc(l2.outputs, sizeof(float));
    }
    // added another layer for predictions
    predictions = (float***) calloc(YOLO, sizeof(float**));
    for (int i = 0; i < YOLO; i++) {
      predictions[i] = (float**) calloc(demo_frame, sizeof(float*));
    }
    
    int j;
    for(j = 0; j < demo_frame; ++j){
         predictions[0][j] = (float *) calloc(l.outputs, sizeof(float));
         predictions[1][j] = (float *) calloc(l2.outputs, sizeof(float));
    }
    // added another layer for boxes
    boxes = (box **)calloc(YOLO, sizeof(box*));
    for (int i = 0; i < YOLO; i++) {
      boxes[0] = (box*) calloc(l.w*l.h*l.n, sizeof(box));
      boxes[1] = (box*) calloc(l2.w*l2.h*l2.n, sizeof(box));
    }

    // added another layer for probs
    probs = (float ***)calloc(YOLO, sizeof(float **));
    for (int i = 0; i < YOLO; i++) {
      probs[0] = (float**) calloc(l.w*l.h*l.n, sizeof(float *));
      probs[1] = (float**) calloc(l2.w*l2.h*l2.n, sizeof(float*));
    }

    for(j = 0; j < l.w*l.h*l.n; ++j){
        probs[0][j] = (float *)calloc(l.classes+1, sizeof(float));
        probs[1][j] = (float *)calloc(l2.classes+1, sizeof(float));
    }

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[1] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[2] = letterbox_image(buff[0], net.w, net.h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    

    pthread_t detect_thread;
    pthread_t detect_thread2;
    pthread_t fetch_thread;
    pthread_t counter_thread; /*added counter thread */

    int count = 0;
    demo_time = get_wall_time();

    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        if(YOLO==2){
        if(pthread_create(&detect_thread2, 0, detect_in_thread2, 0)) error("Thread creation failed");
        }
        #ifdef THREADLINE
        if(pthread_create(&counter_thread, 0, counter_func, 0)) error("Counter Thread creation failed"); /* create the counter thread*/
        #endif
        image im = buff[(buff_index + 1)%3];
        
        if(!prefix){
            fps = 1./(get_wall_time() - demo_time);
            demo_time = get_wall_time();
            #ifdef SAVEVIDEO
            save_video(im, mVideoWriter); /* save the current frame */
            #endif

            draw_detection_in_thread(0);
            /* uncommet to see display */
            //display_in_thread(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            #ifdef SAVEVIDEO
            save_video(im, mVideoWriter); /* save the current frame */
            #else
            save_image(im, name);
            #endif

        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        if(YOLO==2){
        pthread_join(detect_thread2, 0);
        }
        #ifdef THREADLINE
        pthread_join(counter_thread, 0); /* joins the counter_thread back to main process */
        #endif 
        ++count;

    }
    free(result);
}

void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    demo_frame = avg_frames;

    // added another layer for predictions
    predictions = (float***) calloc(YOLO, sizeof(float**));
    for (int i = 0; i < YOLO; i++) {
      predictions[i] = (float**) calloc(demo_frame, sizeof(float*));
    }

    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfg1, weight1, 0);
    set_batch_network(&net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net.layers[net.n-1];
    demo_detections = l.n*l.w*l.h;
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < demo_frame; ++j) predictions[0][j] = (float *) calloc(l.outputs, sizeof(float));

    // added another layer for boxes
    boxes = (box **)calloc(l.w*l.h*l.n, sizeof(box));
    for (int i = 0; i < YOLO; i++) {
      boxes[i] = (box*) calloc(l.w*l.h*l.n, sizeof(box));
    }

    // added another layer for probs
    probs = (float ***)calloc(l.w*l.h*l.n, sizeof(float *));
    for (int i = 0; i < YOLO; i++) {
      probs[i] = (float**) calloc(l.w*l.h*l.n, sizeof(float *));
    }

    for(j = 0; j < l.w*l.h*l.n; ++j) probs[0][j] = (float *)calloc(l.classes+1, sizeof(float));

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[1] = letterbox_image(buff[0], net.w, net.h);
    buff_letter[2] = letterbox_image(buff[0], net.w, net.h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL);
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = get_wall_time();

    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

        if(!prefix){
            fps = 1./(get_wall_time() - demo_time);
            demo_time = get_wall_time();
            display_in_thread(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
