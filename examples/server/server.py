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

# init rtmpose-t
config_file_rtmpose_t = os.path.join(ROOT, 'model/rtmpose-t/rtmpose-t_8xb256-420e_coco-256x192.py')
checkpoint_file_rtmpose_t = os.path.join(ROOT, 'model/rtmpose-t/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth')
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

model_t = init_model(
        config_file_rtmpose_t,
        checkpoint_file_rtmpose_t,
        device='cuda:0',
        cfg_options=cfg_options)

# init visualizer
model_t.cfg.visualizer.radius = 5
model_t.cfg.visualizer.alpha = 0.8
model_t.cfg.visualizer.line_width = 3

# build visualizer
visualizer_t = VISUALIZERS.build(model_t.cfg.visualizer)
visualizer_t.set_dataset_meta(
    model_t.dataset_meta, skeleton_style='mmpose')

# init rtmpose-s
config_file_rtmpose_s = os.path.join(ROOT, 'model/rtmpose-s/rtmpose-s_8xb256-420e_coco-256x192.py')
checkpoint_file_rtmpose_s = os.path.join(ROOT, 'model/rtmpose-s/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth')
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

model_s = init_model(
        config_file_rtmpose_s,
        checkpoint_file_rtmpose_s,
        device='cuda:0',
        cfg_options=cfg_options)

# init visualizer
model_s.cfg.visualizer.radius = 5
model_s.cfg.visualizer.alpha = 0.8
model_s.cfg.visualizer.line_width = 3

# build visualizer
visualizer_s = VISUALIZERS.build(model_s.cfg.visualizer)
visualizer_s.set_dataset_meta(
    model_s.dataset_meta, skeleton_style='mmpose')



class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"
    frame_no = 0
    model = None
    visualizer = None


    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        if self.transform == "RTMPose-s":
            self.model = model_s
            self.visualizer = visualizer_s

    async def recv(self):
        frame = await self.track.recv()
        self.frame_no += 1
        if self.transform in ["RTMPose-s", "RTMPose-t", "RTMPose-m", "RTMPose-l"]:
            # Save the original frame, with the name of frame_no
            # cv2.imwrite(f'test_img_in/frame_{self.frame_no}.jpg', frame.to_ndarray(format="bgr24"))
            # Convert the frame to a numpy array
            img = frame.to_ndarray(format="bgr24")
            # print(img)
            # Resize the image to the desired dimensions
            img_resized = cv2.resize(img, (256, 192))
            img_resized = img

            # Add this code before the pose estimation and visualization code
            start_time = time.time()
            print(f"start")
            # perform pose estimation (inference) on the resized frame(img)
            batch_results = inference_topdown(self.model, img_resized)
            results = merge_data_samples(batch_results)
            pose_estimation_time = time.time()
            print(f"pose estimation time: {pose_estimation_time - start_time}")
            # print results(PoseDataSample) for debugging
            # print(results)

            # Step 1: save the image with the pose estimation results(for validation)
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
