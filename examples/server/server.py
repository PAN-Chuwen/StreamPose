import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import numpy as np
import time

import cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame

# mmpose
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.structures import merge_data_samples
from mmpose.registry import VISUALIZERS


ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

# mmpose init
register_all_modules()


def init_model_and_visualizer(config_file, checkpoint_file):
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

    model = init_model(
        config_file,
        checkpoint_file,
        device='cuda:0',
        cfg_options=cfg_options)

    # init visualizer
    model.cfg.visualizer.radius = 5
    model.cfg.visualizer.alpha = 0.8
    model.cfg.visualizer.line_width = 3

    # build visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style='mmpose')

    return model, visualizer

# init rtmpose-t
config_file_rtmpose_t = os.path.join(ROOT, 'model/rtmpose-t/rtmpose-t_8xb256-420e_coco-256x192.py')
checkpoint_file_rtmpose_t = os.path.join(ROOT, 'model/rtmpose-t/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth')
model_t, visualizer_t = init_model_and_visualizer(config_file_rtmpose_t, checkpoint_file_rtmpose_t)

# init rtmpose-s
config_file_rtmpose_s = os.path.join(ROOT, 'model/rtmpose-s/rtmpose-s_8xb256-420e_coco-256x192.py')
checkpoint_file_rtmpose_s = os.path.join(ROOT, 'model/rtmpose-s/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth')
model_s, visualizer_s = init_model_and_visualizer(config_file_rtmpose_s, checkpoint_file_rtmpose_s)

# init rtmpose-m
config_file_rtmpose_m = os.path.join(ROOT, 'model/rtmpose-m/rtmpose-m_8xb256-420e_coco-256x192.py')
checkpoint_file_rtmpose_m = os.path.join(ROOT, 'model/rtmpose-m/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth')
model_m, visualizer_m = init_model_and_visualizer(config_file_rtmpose_m, checkpoint_file_rtmpose_m)

# init rtmpose-l
config_file_rtmpose_l = os.path.join(ROOT, 'model/rtmpose-l/rtmpose-l_8xb256-420e_coco-256x192.py')
checkpoint_file_rtmpose_l = os.path.join(ROOT, 'model/rtmpose-l/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth')
model_l, visualizer_l = init_model_and_visualizer(config_file_rtmpose_l, checkpoint_file_rtmpose_l)

#init hrnet-32-256x192
config_file_hrnet_256 = os.path.join(ROOT, 'model/hrnet/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192.py')
checkpoint_file_hrnet_256 = os.path.join(ROOT, 'model/hrnet/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192-73ede547_20220914.pth')
model_hrnet_256, visualizer_hrnet_256 = init_model_and_visualizer(config_file_hrnet_256, checkpoint_file_hrnet_256)

#init hrnet-32-384x288
config_file_hrnet_384 = os.path.join(ROOT, 'model/hrnet/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288.py')
checkpoint_file_hrnet_384 = os.path.join(ROOT, 'model/hrnet/td-hm_hrnet-w32_udp-8xb64-210e_coco-384x288-9a3f7c85_20220914.pth')
model_hrnet_384, visualizer_hrnet_384 = init_model_and_visualizer(config_file_hrnet_384, checkpoint_file_hrnet_384)



#init hrnet-32-384x288

models = {
    "RTMPose-s": model_s,
    "RTMPose-t": model_t,
    "RTMPose-m": model_m,
    "RTMPose-l": model_l,
    "hrnet-32-256x192": model_hrnet_256,
    "hrnet-32-384x288": model_hrnet_384,
}

visualizers = {
    "RTMPose-s": visualizer_s,
    "RTMPose-t": visualizer_t,
    "RTMPose-m": visualizer_m,
    "RTMPose-l": visualizer_l,
    "hrnet-32-256x192": visualizer_hrnet_256,
    "hrnet-32-384x288": visualizer_hrnet_384,
}


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"
    frame_no = 0
    model = None
    visualizer = None

    pose_estimation_time_list = []
    recv_called_time = []

    


    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.model = models.get(self.transform)
        self.visualizer = visualizers.get(self.transform)

    async def recv(self):
        current_time = time.time()
        self.recv_called_time.append(current_time)

        # calculate the difference of each element in recv_ called_time when the length reaches 100
        # if(len(self.recv_called_time) == 100):
        #     diff = np.diff(self.recv_called_time)
        #     # calculate the average webRTC transmission time using difference avg
        #     print("average webRTC transmission time: ", (sum(diff) / len(diff)))
        #     # exit the program
        #     exit()

        frame = await self.track.recv()
        self.frame_no = self.frame_no + 1

        if self.transform in ["RTMPose-s", "RTMPose-t", "RTMPose-m", "RTMPose-l", "hrnet-32-256x192", "hrnet-32-384x288"]:
            print("using model", self.transform)
            # Save the original frame, with the name of frame_no
            # cv2.imwrite(f'test_img_in/frame_{self.frame_no}.jpg', frame.to_ndarray(format="bgr24"))
            # Convert the frame to a numpy array
            img = frame.to_ndarray(format="bgr24")

            # TODO: figure out the img resize process, where? in MMPose or in server.py / client.js ?
            # Resize the image to the desired dimensions
            img_resized = img

            # Add this code before the pose estimation and visualization code
            start_time = time.time()
            print(f"start")
            # perform pose estimation (inference) on the resized frame(img)
            batch_results = inference_topdown(self.model, img_resized)
            results = merge_data_samples(batch_results)
            pose_estimation_time = time.time()
            # print the frame rate according to pose estimation time
            print(f"pose estimation time: {pose_estimation_time - start_time}")
            
            #save the time to list
            self.pose_estimation_time_list.append(pose_estimation_time - start_time)

            self.visualizer.add_datasample(
                'result',
                img,
                data_sample=results,
                draw_gt=False,
                draw_bbox=True,
                kpt_thr=0.3,
                draw_heatmap=False,
                show_kpt_idx=False,
                skeleton_style='mmpose',
                show=False,
                out_file=None)
            new_frame_img = self.visualizer.get_image()
            new_frame = VideoFrame.from_ndarray(new_frame_img, format="bgr24")

            # Set the pts and time_base attributes of the new VideoFrame object
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            visualization_time = time.time()
            print(f"visualization time: {visualization_time - pose_estimation_time}")
            return new_frame
        else:
            return frame




async def index(request):
    logging.info("Received request for index")
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def style(request):
    content = open(os.path.join(ROOT, "style.css"), "r").read()
    return web.Response(content_type="text/style", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_get("/style.css", style)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
