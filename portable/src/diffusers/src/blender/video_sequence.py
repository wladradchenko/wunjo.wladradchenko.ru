import os


class VideoSequence:
    def __init__(self,
                 base_dir,
                 frames_path,
                 frame_files_with_interval,  # New parameter
                 input_subdir='media',
                 tmp_subdir='tmp',
                 input_format='frame%04d.jpg',
                 key_format='%04d.jpg',
                 out_subdir_format='out_%s',  # Modified format string
                 blending_out_subdir='blend',
                 output_format='%04d.jpg'):
        self.__base_dir = base_dir
        self.__input_dir = os.path.join(base_dir, input_subdir)
        self.__key_dir = frames_path
        self.__tmp_dir = os.path.join(base_dir, tmp_subdir)
        self.__input_format = input_format
        self.__blending_out_dir = os.path.join(base_dir, blending_out_subdir)
        self.__key_format = key_format
        self.__out_subdir_format = out_subdir_format
        self.__output_format = output_format

        # video frames
        self.__input_frames = [frame_base for frame_base in sorted(os.listdir(self.__input_dir)) if ".png" in frame_base]

        # Store the list of frame files
        self.__frame_files_with_interval = [frame_file for frame_file in frame_files_with_interval if ".png" in frame_file]
        self.__n_seq = len(self.__frame_files_with_interval)

        self.__make_out_dirs()
        os.makedirs(self.__tmp_dir, exist_ok=True)

    @property
    def output_format(self):
        return self.__output_format

    @property
    def frame_files(self):
        return self.__frame_files_with_interval

    @property
    def n_seq(self):
        return self.__n_seq

    @property
    def key_dir(self):
        return self.__key_dir

    @property
    def blending_out_dir(self):
        return self.__blending_out_dir

    def __get_out_subdir(self, frame_name):
        # Use the frame name to create the output directory name
        out_subdir = os.path.join(self.__base_dir, self.__out_subdir_format % frame_name.split(".")[0])
        return out_subdir

    def __get_tmp_out_subdir(self, frame_name):
        tmp_out_subdir = os.path.join(self.__tmp_dir, self.__out_subdir_format % frame_name.split(".")[0])
        return tmp_out_subdir

    def __make_out_dirs(self):
        os.makedirs(self.__base_dir, exist_ok=True)
        os.makedirs(self.__blending_out_dir, exist_ok=True)
        for frame_file in self.__frame_files_with_interval:
            out_subdir = self.__get_out_subdir(frame_file)
            tmp_subdir = self.__get_tmp_out_subdir(frame_file)
            os.makedirs(out_subdir, exist_ok=True)
            os.makedirs(tmp_subdir, exist_ok=True)

        last_input_frame = self.__input_frames[-1]
        out_subdir = self.__get_out_subdir(last_input_frame)
        tmp_subdir = self.__get_tmp_out_subdir(last_input_frame)
        os.makedirs(out_subdir, exist_ok=True)
        os.makedirs(tmp_subdir, exist_ok=True)

    def get_input_sequence(self, i, is_forward=True):
        if i + 1 > len(self.__frame_files_with_interval) - 1:
            # check what file will exist
            last_input_frame = self.__input_frames[-1]
            last_interval_frame = self.__frame_files_with_interval[i]
            if last_input_frame == last_interval_frame:
                return None
            else:
                beg_id = int(last_interval_frame.split(".")[0])
                end_id = int(last_input_frame.split(".")[0])
        else:
            beg_id = self.get_sequence_beg_id(i)
            end_id = self.get_sequence_beg_id(i + 1)
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            id_list = list(range(end_id, beg_id, -1))
        path_dir = [os.path.join(self.__input_dir, self.__input_format % id) for id in id_list if self.__input_format % id in self.__input_frames]
        return path_dir

    def get_output_sequence(self, i, is_forward=True):
        if i + 1 > len(self.__frame_files_with_interval) - 1:
            # check what file will exist
            last_input_frame = self.__input_frames[-1]
            last_interval_frame = self.__frame_files_with_interval[i]
            if last_input_frame == last_interval_frame:
                return None
            else:
                beg_id = int(last_interval_frame.split(".")[0])
                end_id = int(last_input_frame.split(".")[0])
                interval_frame_name = self.__input_frames[-1]
        else:
            beg_id = self.get_sequence_beg_id(i)
            end_id = self.get_sequence_beg_id(i + 1)
            interval_frame_name = self.__frame_files_with_interval[i]
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        out_subdir = self.__get_out_subdir(interval_frame_name)
        path_dir = [os.path.join(out_subdir, self.__output_format % id) for id in id_list if self.__input_format % id in self.__input_frames]
        return path_dir

    def get_flow_sequence(self, i, is_forward=True):
        if i + 1 > len(self.__frame_files_with_interval) - 1:
            # check what file will exist
            last_input_frame = self.__input_frames[-1]
            last_interval_frame = self.__frame_files_with_interval[i]
            if last_input_frame == last_interval_frame:
                return None
            else:
                beg_id = int(last_interval_frame.split(".")[0])
                end_id = int(last_input_frame.split(".")[0])
        else:
            beg_id = self.get_sequence_beg_id(i)
            end_id = self.get_sequence_beg_id(i + 1)
        if is_forward:
            id_list = list(range(beg_id, end_id - 1))
            path_dir = [os.path.join(self.__tmp_dir, 'flow_f_%04d.npy' % id) for id in id_list if self.__input_format % id in self.__input_frames]
        else:
            id_list = list(range(end_id, beg_id + 1, -1))
            path_dir = [os.path.join(self.__tmp_dir, 'flow_b_%04d.npy' % id) for id in id_list if self.__input_format % id in self.__input_frames]

        return path_dir

    def get_edge_sequence(self, i, is_forward=True):
        if i + 1 > len(self.__frame_files_with_interval) - 1:
            # check what file will exist
            last_input_frame = self.__input_frames[-1]
            last_interval_frame = self.__frame_files_with_interval[i]
            if last_input_frame == last_interval_frame:
                return None
            else:
                beg_id = int(last_interval_frame.split(".")[0])
                end_id = int(last_input_frame.split(".")[0])
                interval_frame_name = self.__input_frames[-1]
        else:
            beg_id = self.get_sequence_beg_id(i)
            end_id = self.get_sequence_beg_id(i + 1)
            interval_frame_name = self.__frame_files_with_interval[i]
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        tmp_dir = self.__get_tmp_out_subdir(interval_frame_name)
        path_dir = [os.path.join(tmp_dir, 'edge_' + self.__output_format % id) for id in id_list if self.__input_format % id in self.__input_frames]
        return path_dir

    def get_temporal_sequence(self, i, is_forward=True):
        if i + 1 > len(self.__frame_files_with_interval) - 1:
            # check what file will exist
            last_input_frame = self.__input_frames[-1]
            last_interval_frame = self.__frame_files_with_interval[i]
            if last_input_frame == last_interval_frame:
                return None
            else:
                beg_id = int(last_interval_frame.split(".")[0])
                end_id = int(last_input_frame.split(".")[0])
                interval_frame_name = self.__input_frames[-1]
        else:
            beg_id = self.get_sequence_beg_id(i)
            end_id = self.get_sequence_beg_id(i + 1)
            interval_frame_name = self.__frame_files_with_interval[i]
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        tmp_dir = self.__get_tmp_out_subdir(interval_frame_name)
        path_dir = [os.path.join(tmp_dir, 'temporal_' + self.__output_format % id) for id in id_list if self.__input_format % id in self.__input_frames]
        return path_dir

    def get_pos_sequence(self, i, is_forward=True):
        if i + 1 > len(self.__frame_files_with_interval) - 1:
            # check what file will exist
            last_input_frame = self.__input_frames[-1]
            last_interval_frame = self.__frame_files_with_interval[i]
            if last_input_frame == last_interval_frame:
                return None
            else:
                beg_id = int(last_interval_frame.split(".")[0])
                end_id = int(last_input_frame.split(".")[0])
                interval_frame_name = self.__input_frames[-1]
        else:
            beg_id = self.get_sequence_beg_id(i)
            end_id = self.get_sequence_beg_id(i + 1)
            interval_frame_name = self.__frame_files_with_interval[i]
        if is_forward:
            id_list = list(range(beg_id, end_id))
        else:
            i += 1
            id_list = list(range(end_id, beg_id, -1))
        tmp_dir = self.__get_tmp_out_subdir(interval_frame_name)
        path_dir = [os.path.join(tmp_dir, 'pos_' + self.__output_format % id) for id in id_list if self.__input_format % id in self.__input_frames]
        return path_dir

    def get_sequence_beg_id(self, i):
        return int(self.__frame_files_with_interval[i].split(".")[0])

    def get_blending_img(self, i):
        return os.path.join(self.__blending_out_dir, self.__output_format % i)
