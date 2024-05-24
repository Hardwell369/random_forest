"""randomforest package.

randomforest
"""

import structlog
from bigmodule import I

# 需要安装的第三方依赖包
# from bigmodule import R
# R.require("requests>=2.0", "isort==5.13.2")

# metadata
# 模块作者
author = "BigQuant"
# 模块分类
category = "机器学习"
# 模块显示名
friendly_name = "RandomForest"
# 文档地址, optional
doc_url = "https://bigquant.com/wiki/"
# 是否自动缓存结果
cacheable = True

def _train(x, y, n_estimators=20, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=0.9, random=0, render_chart=True):
    """
    :params x, y: 特征列(DataFrame), 标签列
    :params n_estimators: 树的种类
    :params criterion: 评估函数
    :params max_depth: 每棵树的最大深度
    :params min_samples_split: 分割节点所需的最小样本数
    :params min_samples_leaf: 分出叶子节点后每个子节点所包含的最小样本数
    :params max_features: 每次分支时只考虑在max_features * feature_nums数量上进行搜索
    :params render_chart: 是否绘制柱状图
    """
    import os
    import subprocess

    import dai
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # 初始化日志
    logger = structlog.get_logger()

    if criterion in ["gini", "entropy", "log_loss"]:
        model = RandomForestClassifier(n_estimators=n_estimators, 
                                        criterion=criterion, 
                                        max_depth=max_depth, 
                                        min_samples_split=min_samples_split, 
                                        min_samples_leaf=min_samples_leaf, 
                                        max_features=max_features)
        logger.info("初始化一个随机森林分类器")
    
    elif criterion in ["squared_error", "absolute_error", "friedman_mse", "poisson"]:
        model = RandomForestRegressor(n_estimators=n_estimators, 
                                        criterion=criterion, 
                                        max_depth=max_depth, 
                                        min_samples_split=min_samples_split, 
                                        min_samples_leaf=min_samples_leaf, 
                                        max_features=max_features)
        logger.info("初始化一个随机森林回归器")
    
    # 模型拟合
    model.fit(x, y)

    return {'model': model}

def _train_cache(input_1, input_2, input_3):
    """
    模型缓存
    :params input_1: 传入数据x
    :params input_2: 传入模型
    """
    import dai
    x_ds = input_1
    x = x_ds.read()
    y = x['label']
    x = x.drop(['label'], axis=1)
    n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, random, render_chart = input_2
    model = _train(x, y, n_estimators, criterion.split('(')[0], max_depth, min_samples_split, min_samples_leaf, max_features, random, render_chart)
    return dict(model=dai.DataSource.write_pickle(model))

def _predict_cache(input_1, input_2, input_3):
    """
    预测结果缓存
    """
    import dai
    model = input_2.read()
    pred = model.predict(input_1.read())
    return dict(pred=dai.DataSource.write_pickle({'pred': pred}))

def run(
    input_1: I.port("训练数据: 标签名称必须是label", optional=True) = None, 
    input_2: I.port("预测数据", optional=True) = None, 
    input_3: I.port('加载模型', optional=True) = None,

    n_estimators: I.int("树的数量") = 10, 
    criterion: I.choice("评估函数", values=['squared_error(回归)', 'absolute_error(回归)', 'friedman_mse(回归)', 'poisson(回归)', 'gini(分类)', 'entropy(分类)', 'log_loss(分类)'])='squared_error(回归)', 
    max_depth: I.int("树的最大深度", min = 1, max = 100) = 20,
    min_samples_split: I.int("分割节点所需的最小样本数", min = 1, max = 10) = 2, 
    min_samples_leaf: I.int("分枝后每个子节点所包含最小样本数", min = 1, max = 10) = 2, 
    max_features: I.float("特征利用率", min=0.1, max=1.0) = 0.9, 
    random:  I.int("随机种子") = 0, 
    render_chart: I.bool("是否显示特征重要性图标") = True
)->[
    I.port("预测数据", "prediction"), 
    I.port("固化的模型", "model")
]:
    import dai
    from bigmodule import M

    # 训练
    if input_3:
        model_dict = input_3.read()
    elif input_1:
    # 加载训练集
        train_df = input_1.read()
        feature_df = train_df.drop(['date', 'instrument', 'label'], axis=1)
        label = train_df['label']
        x = feature_df
        x['label'] = label

        x_ds = dai.DataSource.write_bdb(x)
        model_dict = M.python.v1(run=_train_cache, 
                                input_1=(x_ds), 
                                input_2=(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, random, render_chart)).model.read()

    # 预测
    test_df = input_2.read()
    feature = test_df.drop(['date', 'instrument'], axis=1)
    columns = list(feature.columns)                                # 存下它的字段名
    feature = dai.DataSource.write_pickle(feature)
    model = dai.DataSource.write_pickle(model_dict['model'])
    ypre = M.python.v1(run=_predict_cache, 
                                input_1=(feature), 
                                input_2=(model)).pred.read()['pred']
    test_df['pre_label'] = ypre

    # 还需要打包模型
    model = {'model': model_dict['model']}
    return I.Outputs(prediction=dai.DataSource.write_bdb(test_df[['date', 'instrument', 'pre_label']]), 
                    model=dai.DataSource.write_pickle(model), 
                    columns_name = dai.DataSource.write_pickle({'columns': columns}), 
                    plot_chart=render_chart)


def post_run(outputs):
    """
    后置运行函数
    """
    import bigcharts
    import pandas as pd

    # 特征重要性可视化配置
    if outputs.plot_chart: 
        # 加载模型
        model = outputs.model.read()['model']

        # 标记特征重要性
        feature_importances = pd.DataFrame([])
        feature_importances['name'] = outputs.columns_name.read()['columns']
        feature_importances['feature_importance'] = model.feature_importances_
        feature_importances = feature_importances.sort_values('feature_importance', ascending=False).head(20)

        # 可视化
        xaxis_opts = bigcharts.opts.AxisOpts(axislabel_opts=bigcharts.opts.LabelOpts(rotate=45), splitline_opts=bigcharts.opts.SplitLineOpts(is_show=False))
        bar = bigcharts.Bar(data=(feature_importances, 'name', ['feature_importance']), title="特征重要性", datazoom=False, init_opts=bigcharts.opts.InitOpts(width="100%"))
        bar.options["xAxis"][0].update(xaxis_opts.opts)
        bar.render()
    return outputs
