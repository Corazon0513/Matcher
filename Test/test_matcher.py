# -*-coding:utf-8-*-
import os

from matcher import Matcher
"""
用于检测广告元素是否正确显示。
判断待测图片中是否存在模版图片。
待测图片中的目标与模版可以是缩放，平移关系。
不可以是旋转，扭曲关系。
基于OpenCV的matchTemplate方法实现。
"""


def match_test(src, tmp, is_pass, is_verbose=False):
    """
    模版测试-正常模版
    """
    print('\n[TEST]AdMatch类测试-模版:' + tmp)
    # ad_match = AdMatch(
    #     src,
    #     tmp,
    #     is_debug=verbose,
    #     count_time=2,
    # )
    # ad_match.process()
    # print("[TEST]检测总耗时：" + str(ad_match.total_run_time) + "s.")
    # if ad_match.match_judge == is_pass:
    #     print("[TEST]目标：" + src + " 测试通过！")
    #     if is_verbose:
    #         ad_match.show(need_match_raw_result=False)
    #     return True
    # else:
    #     print("[TEST]目标：" + src + " 测试失败！")
    #     if is_verbose:
    #         ad_match.show(need_match_raw_result=False)
    #     return False
    match = Matcher(src, tmp).run_immediate()
    if match == is_pass:
        print("[TEST]目标：" + src + " 测试通过！")
        return True
    else:
        print("[TEST]目标：" + src + " 测试失败！")
        return False


if __name__ == '__main__':
    # 启动啰嗦模式，且在用例测试失败时显示图片。
    verbose = False
    # 待测文件路径
    filepath = '../assets/65/'
    # 模版路径
    tmp_file = filepath + 'ad_tmp.png'
    # 匹配用例列表
    pass_files = []
    # 不匹配用例列表
    reject_files = []
    # 分类正确率统计
    accuracy = {'Correct case': 0,
                'Total case': 0,
                'Accuracy rate': 1}
    # 用例字典
    test_case_files = {'pass_files': pass_files, 'reject_files': reject_files}
    # 枚举并保存用例
    for f in os.listdir(filepath + 'pass/'):
        if 'png' in f:
            test_case_files['pass_files'].append('%s%s' % (filepath + 'pass/', f))
    for f in os.listdir(filepath + 'reject/'):
        if 'png' in f:
            test_case_files['reject_files'].append('%s%s' % (filepath + 'reject/', f))
    # 统计并保存用例总数
    accuracy['Total case'] = len(test_case_files['pass_files']) + len(test_case_files['reject_files'])
    # 测试匹配的用例
    for test_case in test_case_files['pass_files']:
        if match_test(test_case, tmp_file, True, is_verbose=verbose):
            accuracy['Correct case'] += 1
    # 测试不匹配的用例
    for test_case in test_case_files['reject_files']:
        if match_test(test_case, tmp_file, False, is_verbose=verbose):
            accuracy['Correct case'] += 1
    # 计算用例通过率
    accuracy['Accuracy rate'] = float(accuracy['Correct case']) / float(accuracy['Total case'])
    print('\n[TEST]测试完成，正确率：' + str(accuracy['Accuracy rate'] * 100) + '%')
    pass
