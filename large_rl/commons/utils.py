""" All miscellaneous util fns """
import re
import math
import datetime
import numpy as np
import os
import os.path as osp

try:
    import moviepy.editor as mpy
    import cv2
    from large_rl.envs.mining.minigrid import CELL_PIXELS
except:
    pass


class wandb_logger(object):
    def __init__(self, args: dict):
        self.wandb = None
        self._create_prefix(args=args)
        if args["wand"]:
            self.wandb = self._create_wandb(args=args)

    def _create_wandb(self, args: dict):
        import wandb
        from wandb_credentials import WANDB_API_KEY
        wandb.login(key=WANDB_API_KEY)

        if args["group_name"] is None:
            args["group_name"] = args["prefix"]

        wandb.init(settings=dict(start_method='thread'), project='rl-recsys-analysis', entity='clvr', name=self.prefix, group=args["group_name"], dir=args["wandb_dir"])
        wandb.config.update(args)
        args["prefix"] = self.prefix
        return wandb

    def wandb_log_image(self, images: list, step: int, log_string):
        if self.wandb is not None:
            self.wandb.log({log_string: [self.wandb.Image(image) for image in images]}, step=step)

    def wandb_log_video(self, path_to_video: str, step: int, log_string, format="gif"):
        if self.wandb is not None:
            self.wandb.log(
                {log_string: self.wandb.Video(path_to_video, fps=4, format=format)})  # doesn't support step=step

    def wandb_log(self, data: dict, step: int):
        if self.wandb is not None:
            self.wandb.log(data, step=step)

    def _create_prefix(self, args: dict):
        assert args["prefix"] is not None and args["prefix"] != '', 'Must specify a prefix to use W&B'
        d = datetime.datetime.today()
        date_id = f"{d.month}{d.day}{d.hour}{d.minute}{d.second}"
        before = f"{date_id}-{args['seed']}-"

        if args["prefix"] != 'debug' and args["prefix"] != 'NONE':
            self.prefix = before + args["prefix"]
            print('Assigning full prefix %s' % self.prefix)
        else:
            self.prefix = args["prefix"]


def logging(*msg):
    # def prRed(prt): print("\033[91m {}\033[00m".format(prt))
    # def prGreen(prt): print("\033[92m {}\033[00m".format(prt))
    # def prYellow(prt): print("\033[93m {}\033[00m".format(prt))
    # def prLightPurple(prt): print("\033[94m {}\033[00m".format(prt))
    # def prPurple(prt): print("\033[95m {}\033[00m".format(prt))
    # def prCyan(prt): print("\033[96m {}\033[00m".format(prt))
    # def prLightGray(prt): print("\033[97m {}\033[00m".format(prt))
    # def prBlack(prt): print("\033[98m {}\033[00m".format(prt))

    print("{}>".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), *msg)


def _test_logging():
    print("=== test logging ===")
    logging("a"
            "b"
            "c")


def mean_dict(_list_dict: list):
    result = {}
    for d in _list_dict:
        for k in d.keys():
            result[k] = result.get(k, 0) + d[k]

    for k, v in result.items():
        result[k] = float(v) / float(len(_list_dict))
    return result


def softmax(_vec):
    """Computes the softmax of a vector."""
    normalized_vector = np.array(_vec) - np.max(_vec)  # For numerical stability
    return np.exp(normalized_vector) / np.sum(np.exp(normalized_vector))


def min_max_scale_vec(_vec: np.ndarray, _min: float, _max: float):
    _num = (_vec - np.min(_vec))
    _den = (np.max(_vec) - np.min(_vec))
    _fraction = _num / _den if _den else 0.0  # Safe division!
    return _fraction * (_max - _min) + _min


def scale_number(x, to_min, to_max, from_min, from_max):
    return (to_max - to_min) * (x - from_min) / (from_max - from_min) + to_min


def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def softplus_math(x):
    return math.log1p(math.exp(-abs(x))) + max(x, 0)


def scaled_sigmoid(x, _min, _max):
    return _min + (_max - _min) / (1 + math.exp(-x))


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def _test_mean_dict():
    print("=== _test_mean_dict ===")
    _k = 11
    _list_dict = [dict(a=i * 1, b=i * 2, c=i * 3) for i in range(1, _k)]
    res = mean_dict(_list_dict=_list_dict)
    print(res)

    _list_dict = [{'hit_rate': 0.0, 'mrr': 0.0, 'ndcg': 0.0} for _ in range(10)]
    res = mean_dict(_list_dict=_list_dict)
    print(res)

    _list_dict = [{'hit_rate': 0.5, 'mrr': 0.5, 'ndcg': 0.5} for _ in range(1)]
    res = mean_dict(_list_dict=_list_dict)
    print(res)


def _test_min_max_scale():
    # print(min_max_scale(x=0.0, _min=0.0, _max=1.0))
    # print(min_max_scale(x=-0.0, _min=0.0, _max=1.0))
    # print(min_max_scale(x=-0.0001, _min=0.0, _max=1.0))

    print(min_max_scale_vec(_vec=np.asarray([-1.0, 0.1, 1.0]), _min=0.0, _max=2.0))
    print(min_max_scale_vec(_vec=np.asarray([-1.0, 0.5, 1.0]), _min=0.0, _max=2.0))
    print(min_max_scale_vec(_vec=np.asarray([-1.0, 0.8, 1.0]), _min=0.0, _max=2.0))

    # print(min_max_scale_vec(_vec=np.asarray([0.0, 0.1, 2.0]), _min=0.0, _max=2.0))
    # print(min_max_scale_vec(_vec=np.asarray([0.0, 0.5, 2.0]), _min=0.0, _max=2.0))
    # print(min_max_scale_vec(_vec=np.asarray([0.0, 0.8, 2.0]), _min=0.0, _max=2.0))


def _test_scaling_num():
    print(scale_number(x=0.2, to_min=0.0, to_max=1.0, from_min=-1.0, from_max=1.0))


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)

    # NOTE: Added to support renderings that return images series.
    if len(img_nhwc.shape) == 1:
        # Pad other sequences so they are as long.
        dims = [img_nhwc[i].shape[0] for i in range(img_nhwc.shape[0])]
        max_dim = max(dims)
        for i in range(img_nhwc.shape[0]):
            if img_nhwc[i].shape[0] < max_dim:
                img_nhwc[i] = np.pad(img_nhwc[i], ((0, 1), (0, 0), (0, 0), (0, 0)), 'edge')
        img_nhwc = np.array(list(img_nhwc))
        return tile_images(img_nhwc)

    if len(img_nhwc.shape) == 5:
        result = np.array([tile_images(img_nhwc[:, i]) for i in range(img_nhwc.shape[1])])
        return result

    N, h, w, c = img_nhwc.shape

    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    # add boundary to distinguish environments
    img_HWhwc[:, :, :, -1, :] = 0
    img_HWhwc[:, :, -1, :, :] = 0
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def save_mp4(frames, vid_dir, name, fps=10.0, no_frame_drop=False):
    frames = np.array(frames)
    if len(frames[0].shape) == 4:
        new_frames = frames[0]
        for i in range(len(frames) - 1):
            new_frames = np.concatenate([new_frames, frames[i + 1]])
        frames = new_frames

    if no_frame_drop:
        def f(t):
            idx = min(int(t * fps), len(frames) - 1)
            return frames[idx]

        if not osp.exists(vid_dir):
            os.makedirs(vid_dir)

        vid_file = osp.join(vid_dir, name + '.mp4')
        if osp.exists(vid_file):
            os.remove(vid_file)

        video = mpy.VideoClip(f, duration=len(frames) / fps)
        video.write_videofile(vid_file, fps, verbose=False, logger=None)

    else:
        drop_frame = 1.5

        def f(t):
            frame_length = len(frames)
            new_fps = 1. / (1. / fps + 1. / frame_length)
            idx = min(int(t * new_fps), frame_length - 1)
            return frames[int(drop_frame * idx)]

        if not osp.exists(vid_dir):
            os.makedirs(vid_dir)

        vid_file = osp.join(vid_dir, name + '.mp4')
        if osp.exists(vid_file):
            os.remove(vid_file)

        video = mpy.VideoClip(f, duration=len(frames) / fps / drop_frame)
        video.write_videofile(vid_file, fps, verbose=False, logger=None)


class VideoFrameBuffer(object):
    def __init__(self, args, action_meaning_dict: dict):
        self.args = args
        self.repeat_num = 1
        self.num_envs = args['num_envs']
        self.raw_buffer = list()
        self.raw_list_action = list()
        self.raw_multi_opt_bonus = list()
        self.action_meaning_dict = action_meaning_dict
        self.done_cnt = np.zeros(self.num_envs)

    def append(self, frames, list_action=None, done_mask=None, can_multi_opt_bonus=None):
        if done_mask is None:
            done_mask = np.zeros(self.num_envs)
        for i, done in enumerate(done_mask):
            if done:
                if self.done_cnt[i] > 1:
                    frames[i] = self.raw_buffer[-1][i]
                    if list_action is not None:
                        list_action[i] = self.raw_list_action[-1][i]
                else:
                    self.done_cnt[i] += 1
        self.raw_buffer.append(frames)
        if list_action is not None:
            self.raw_list_action.append(list_action)
        if can_multi_opt_bonus is not None:
            self.raw_multi_opt_bonus.append(can_multi_opt_bonus)

    def append_dark(self):
        if len(self.raw_buffer) == 0:
            return
        dark = np.zeros(self.raw_buffer[-1][0].shape).astype(int)
        for _ in range(self.repeat_num):
            self.raw_buffer.append([dark for _ in range(self.num_envs)])

    @property
    def empty(self):
        return len(self.raw_buffer) == 0

    def gen_video(self, test=True):
        # if self.args["if_debug"]:
        #     import pudb; pudb.start()
        frame_buffer = list()
        for _ts in range(len(self.raw_buffer)):
            imgs = self.raw_buffer[_ts]
            if self.args["mw_video_append_action_candidate"]:
                # self.raw_buffer contains some offset at the end so that we need to avoid it
                if len(self.raw_list_action) > 0 and _ts < (len(self.raw_buffer) - 1 - self.repeat_num):
                    for env_id, img in enumerate(imgs):
                        # Add action candidate list
                        action_id_list = self.raw_list_action[_ts][env_id].flatten().tolist()
                        if -1 in action_id_list:
                            img = cv2.putText(img=img, text=f"list: None",
                                              org=(10, 140),
                                              fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=CELL_PIXELS / 60,
                                              color=(128, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                        else:
                            action_meaning_list = []
                            for i in range(len(action_id_list)):
                                action_meaning_list.append(self.action_meaning_dict[action_id_list[i]])
                            img = cv2.putText(img=img, text=f"list: {str(action_meaning_list)}",
                                              org=(10, 140),
                                              fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=CELL_PIXELS / 60,
                                              color=(128, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            frame = tile_images(imgs)
            for _ in range(self.repeat_num):
                frame_buffer.append(frame)
            if _ts == len(self.raw_buffer) - 1:
                for _ in range(self.repeat_num):
                    frame_buffer.append(frame)
        return frame_buffer


if __name__ == '__main__':
    _test_min_max_scale()
    # _test_logging()
    # _test_mean_dict()
