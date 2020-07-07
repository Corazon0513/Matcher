# -*-coding:utf-8-*-

"""
图片模版匹配类
Author: corazonwang@tencent.com
"""

import cv2 as cv
import time
import numpy as np


class Matcher:
    """
    模版检测类
    参数：
        (string)src_path:待测图片路径或ndarray(RGB三通道)
        (string)tmp_path:模版图片路径或ndarray(RGB三通道)
        (bool)is_debug:调试模式，开启图片输出及调试信息输出
        (bool)count_time:输出耗时统计信息
        (float)match_threshold:匹配结果判定阈值
        (float)blur_threshold:模糊度匹配结果判定阈值
    """

    class TimeCount:
        """
        用于检测代码段运行时间
        参数：
            msg:(string)本次计时的描述信息
            total_run_time:(list)累加计时，将本次计时值累加进list[0]
            enable:(int) 0->禁用计时
                         1->启用计时，不启用信息输出
                         2->启用计时，启用信息输出
        """

        def __init__(self, msg, total_run_time, enable=0):
            self._enable = enable
            if self._enable > 0:
                self._msg = msg
                self._start = 0
                self._stop = 0
                self._total = 0
                self._total_run_time = total_run_time
            else:
                pass

        def __enter__(self):
            if self._enable:
                self._start = time.clock()
            else:
                pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._enable:
                self._stop = time.clock()
                self._total = self._stop - self._start
                self._total_run_time[0] += self._total
                if self._enable > 1:
                    print("[TIME]" + self._msg + "耗时:" + str(self._total) + "s.")
            else:
                pass

    def __init__(self, source, temple, is_debug=False,
                 count_time=0, match_threshold=0.8, blur_threshold=0.7):
        self._total_run_time = [0.0]
        with self.TimeCount("初始化", self._total_run_time, enable=count_time):
            self._match_threshold = match_threshold
            self._count_time = count_time
            self._is_debug = is_debug
            self._src = self._load_input(source, describe="目标")
            self._tmp = self._load_input(temple, describe="模版")
            self._debug_print("待测图片分辨率:(%dx%d), 模版图片分辨率:(%dx%d)"
                              % (self._src.shape[0], self._src.shape[1],
                                 self._tmp.shape[0], self._tmp.shape[1]))
            # 确认模版长宽不超过待测图片，否则降低模版分辨率
            self._negative_dimension_handle()
            # 匹配结果保存Mat
            self._result = None
            # 最佳待测图片缩小比率
            self._best_shrink_rate = None
            if self._is_debug:
                self._dst = self._load_input(source, describe="展示",
                                             imread_color=cv.IMREAD_COLOR,
                                             cvtColor_color=None)
            # 匹配点信息
            self._match_val = None
            self._match_loc = None
            self._match_point_info = [0.0, 0.0, (0, 0), (0, 0)]
            # 模版清晰度得分
            self._blur_detect_tmp_score = None
            # 目标清晰度得分
            self._blur_detect_src_score = None
            # 模糊度匹配结果判定阈值
            self._blur_threshold = blur_threshold

    def process(self, match_method=cv.TM_CCOEFF_NORMED):
        """
        进行模版匹配处理
        参数:
            match_method:匹配算法，选项：
                TM_SQDIFF - 平方差匹配法
                TM_SQDIFF_NORMED - 归一化平方差匹配法
                TM_CCORR - 相关匹配法
                TM_CCORR_NORMED - 归一化相关匹配法
                TM_CCOEFF - 系数匹配法
                TM_CCOEFF_NORMED - 归一化系数匹配法
        """
        with self.TimeCount("图像匹配", self._total_run_time, enable=self._count_time):
            # 循环匹配模版
            self._looping_match_template(match_method)
            # 清晰度检测
            self._blur_detect()
            self._debug_print("匹配结果最小值: " + str(round(self._match_point_info[0], 4)) +
                  " @" + str(self._match_point_info[2])
                  + ", 匹配结果最大值:" + str(round(self._match_point_info[1], 4)) +
                  " @" + str(self._match_point_info[3]))
            if self._is_debug:
                self._draw_match_area(self._dst)

    def _looping_match_template(self, match_method, shrink_step=0.1, _show_process=True):
        # 循环缩小目标，匹配模版
        shrink_rate = 1.0
        match_result_list = []
        while (shrink_rate > 0 and
               self._src.shape[0] * shrink_rate > self._tmp.shape[0] and
               self._src.shape[1] * shrink_rate > self._tmp.shape[1]):
            # 缩小目标
            src_resized = cv.resize(self._src, (int(self._src.shape[1] * shrink_rate),
                                                int(self._src.shape[0] * shrink_rate)))
            # 匹配模版
            self._result = cv.matchTemplate(src_resized, self._tmp, match_method)
            # 添加本次匹配结果
            match_result_list.append((shrink_rate, cv.minMaxLoc(self._result)))
            # 更新下次缩小比率
            shrink_rate *= (1 - shrink_step)
        # 提取匹配值列表
        match_val_list = []
        for _ in match_result_list:
            if match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED:
                # 平方差算法 取最小值
                match_val_list.append(_[1][0])
            else:
                # 其他算法 取最大值
                match_val_list.append(_[1][1])
        # 寻找最好的匹配缩放比率
        self._best_shrink_rate = match_result_list[match_val_list.index(max(match_val_list))][0]
        # 获取最好的匹配缩放比率下的匹配信息
        min_val, max_val, min_loc, max_loc = match_result_list[match_val_list.index(max(match_val_list))][1]
        # 根据匹配算法的不同，选择值最小点或值最大点作为匹配点
        if match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED:
            self._match_loc = min_loc
            self._match_val = min_val
        else:
            self._match_loc = max_loc
            self._match_val = max_val
        # 将目标图缩放为最佳匹配尺寸
        self._src = cv.resize(self._src, (int(self._src.shape[1] * self._best_shrink_rate),
                                          int(self._src.shape[0] * self._best_shrink_rate)))
        if self._is_debug:
            # 将展示图缩放为最佳匹配尺寸
            self._dst = cv.resize(self._dst, (int(self._dst.shape[1] * self._best_shrink_rate),
                                              int(self._dst.shape[0] * self._best_shrink_rate)))
        self._debug_print("缩放步进" + str(shrink_step) + ",尝试缩放" + str(len(match_val_list))
                          + "次, 最佳图片缩放" + str(self._best_shrink_rate) + "倍。")
        self._match_point_info = [min_val, max_val, min_loc, max_loc]

    def show(self, need_match_raw_result=False):
        """
        输出图片
        """
        if self._is_debug:
            with self.TimeCount("显示图像", self._total_run_time, enable=self._count_time):
                cv.imshow('Src', self._dst)
                cv.imshow('Tmp', self._tmp)
                if need_match_raw_result:
                    cv.imshow('result', self._result)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("需要is_debug=True才能可视化结果")

    @property
    def match_info(self):
        """
        返回匹配信息
        get_match_info() -> (tuple)locate, (float)value
        """
        return self._match_loc, self._match_val

    @property
    def match_judge(self):
        """
        返回匹配判定
        get_match_info() -> (bool)is_pass
        """
        # 防止未执行process就读取判定结果
        if self._match_val is None:
            raise RuntimeError("在图片分析之前读取了分析结果。")
        # 检测匹配度
        if self._match_val >= self._match_threshold:
            pass
        else:
            self._debug_print("匹配失败：未找到阈值足够高的匹配区域")
            return False
        # 检测模糊度
        if self._blur_detect_src_score / self._blur_detect_tmp_score > self._blur_threshold:
            pass
        else:
            self._debug_print("匹配失败：匹配区域清晰度不足")
            return False
        # 匹配成功
        return True

    @property
    def total_run_time(self):
        """
        返回总运行耗时
        get_total_run_time() -> (float)time
        """
        return self._total_run_time[0]

    def _negative_dimension_handle(self):
        """
        处理模版大于待测图片导致的negative_dimension异常
        """
        while self._src.shape[0] - self._tmp.shape[0] < 0 or self._src.shape[1] - self._tmp.shape[1] < 0:
            self._tmp = cv.resize(self._tmp, (self._tmp.shape[1] / 2, self._tmp.shape[0] / 2))

    def _blur_detect(self, method=1):
        """
        检测目标图片清晰度是否达标
        参数:
            method:(int) 0 -> Tenengrad梯度法
                         1 -> Laplacian梯度法
        """
        with self.TimeCount("清晰度检测", self._total_run_time, enable=self._count_time):
            if self._blur_detect_tmp_score is None:
                if method == 0:
                    dst = cv.Sobel(self._tmp, cv.CV_8U, 1, 1)
                else:
                    dst = cv.Laplacian(self._tmp, cv.CV_8U)
                self._blur_detect_tmp_score = cv.mean(dst)[0]
                self._debug_print("模版清晰度得分:" + str(self._blur_detect_tmp_score))
            src_cut = self._src[self._match_loc[1]:
                                (self._match_loc[1] + self._tmp.shape[0]),
                      self._match_loc[0]:
                      (self._match_loc[0] + self._tmp.shape[1])]
            if method == 0:
                dst = cv.Sobel(src_cut, cv.CV_8U, 1, 1)
            else:
                dst = cv.Laplacian(src_cut, cv.CV_8U)
            self._blur_detect_src_score = cv.mean(dst)[0]
            self._debug_print("识别区清晰度得分:" + str(self._blur_detect_src_score))
            self._debug_print("清晰度比率:" + str(self._blur_detect_src_score / self._blur_detect_tmp_score))
        pass

    def _draw_match_area(self, dst):
        """
        绘制匹配框
        """
        # 根据匹配程度计算颜色(由红至绿匹配度由低至高)
        rectangle_color = (0, int(255.0 * self._match_point_info[1]), int(255.0 * (1 - self._match_point_info[1])))
        # 绘制匹配框，打印匹配度
        cv.rectangle(dst, self._match_loc, (self._match_loc[0] + self._tmp.shape[1],
                                            self._match_loc[1] + self._tmp.shape[0]),
                     rectangle_color, thickness=5)
        cv.putText(dst, str(round(self._match_val, 4)),
                   (self._match_loc[0] + self._tmp.shape[1] / 3, self._match_loc[1] + self._tmp.shape[0] / 2),
                   cv.FONT_HERSHEY_TRIPLEX, 2, rectangle_color, thickness=4)

    def _load_input(self, source, describe=None,
                    imread_color=cv.IMREAD_GRAYSCALE,
                    cvtColor_color=cv.COLOR_RGB2GRAY):
        """
        带类型判断的图片读取
        """
        buff = None
        # 读取图片
        if isinstance(source, str):
            self._debug_print(describe + "图片是以路径形式提供")
            try:
                buff = cv.imread(source, imread_color)
            except Exception as e:
                print(describe + "图片加载失败，请检查文件路径是否正确。")
            finally:
                pass
        elif isinstance(source, np.ndarray):
            self._debug_print(describe + "图片是以numpy数组形式提供")
            if cvtColor_color is None:
                buff = source.copy()
            else:
                buff = cv.cvtColor(source, cvtColor_color)
        else:
            raise AssertionError("未知的" + describe + "图片参数类型")
        return buff

    def _debug_print(self, msg):
        if self._is_debug:
            print("[DEBUG]" + msg)

    def run_immediate(self):
        self.process()
        return self.match_judge
