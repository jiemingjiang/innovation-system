"""Pure, testable analysis functions. No network requests or hidden model calls."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INDEX = '区域创新生态指数'
DIMS = ['创新主体指数', '资源配置指数', '要素流动指数', '创新韧性指数']
FUND = '人民币百万_插值最终结果'
CORE = {'上海市', '南京市', '杭州市', '苏州市', '宁波市'}
COEFS = {'全样本': (0.0045, 0.0015), '核心城市': (-0.0011, 0.0014),
         '非核心城市': (0.0040, 0.0010), '高市场化组': (0.0029, 0.0015),
         '低市场化组': (0.0042, 0.0018)}

def load_data(source=None):
    source = source or ROOT / 'data/source.xlsx'
    with pd.ExcelFile(source) as book:
        raw = pd.read_excel(book, sheet_name='最终结果_插值+指数')
    required = ['城市','年份',INDEX,*DIMS,FUND,'产学研合作','技术合同成交额（万元）']
    missing = set(required)-set(raw.columns)
    if missing:
        raise ValueError('缺少必需字段：'+'、'.join(sorted(missing)))
    raw = raw.dropna(subset=['城市','年份']).copy()
    raw['城市'] = raw['城市'].astype(str).str.strip()
    for col in required[1:]:
        raw[col] = pd.to_numeric(raw[col], errors='raise')
        if not np.isfinite(raw[col]).all():
            raise ValueError(f'{col}存在缺失或非有限数值，请先修正。')
    if (raw['年份'] % 1 != 0).any():
        raise ValueError('年份必须为整数。')
    raw['年份'] = raw['年份'].astype(int)
    if raw.duplicated(['城市','年份']).any():
        raise ValueError('存在重复城市—年份，已拒绝导入。')
    if (raw[[FUND,'产学研合作','技术合同成交额（万元）']] < 0).any().any():
        raise ValueError('基金规模、专利数和技术合同成交额不能为负数。')
    if ((raw[[INDEX,*DIMS]]<0)|(raw[[INDEX,*DIMS]]>1)).any().any():
        raise ValueError('指数不在0—1范围，请核对数据口径。')
    raw['城市层级'] = np.where(raw['城市'].isin(CORE),'核心城市','非核心城市')
    raw['基金规模（亿元，单位换算）'] = raw[FUND]/100
    return raw.sort_values(['城市','年份']).reset_index(drop=True)

def scenario(fund, growth, beta, se, year):
    """Conditional change, holding covariates/FE constant. fund in paper units."""
    if not all(math.isfinite(float(x)) for x in [fund,growth,beta,se,year]):
        raise ValueError('情景参数必须为有限数。')
    if fund < 0 or not 0 <= growth <= 1 or se < 0:
        raise ValueError('基金须非负，增长率须为0—1，标准误须非负。')
    dx = (math.log1p(fund*(1+growth))-math.log1p(fund)) if year >= 2015 else 0.0
    return {'dx':dx, 'delta':beta*dx, 'lower':(beta-1.96*se)*dx,
            'upper':(beta+1.96*se)*dx, 'fund_after':fund*(1+growth)}

def diagnose(df, city, year):
    peer = df.loc[df['年份']==year].set_index('城市')
    row = peer.loc[city]
    percentiles = peer[DIMS].rank(pct=True,method='average')*100
    weak = percentiles.loc[city].idxmin()
    actions = {
        '创新主体指数':('创新主体培育','梳理科技型企业融资需求和高校成果转化项目，优先核查真实融资缺口。','企业存续率、新增高新企业数、社会资本跟投情况'),
        '资源配置指数':('研发资源配置','核查研发人才和科技投入匹配情况；原文资源配置维度系数不显著，不承诺基金扩张能改善该项。','研发人员变化、企业研发强度、项目执行率'),
        '要素流动指数':('技术转化服务','对接技术交易与成果转化服务，先核查交易转化环节的障碍。','技术合同履约率、成果转化周期、技术交易额'),
        '创新韧性指数':('长期金融支持','排查长期融资连续性与数字金融服务覆盖，研究分阶段投资及退出安排。','融资续接率、数字服务覆盖、长期资本占比')}
    title,action,monitor = actions[weak]
    return {'city':city,'year':int(year),'index':float(row[INDEX]),
            'rank':int(peer[INDEX].rank(ascending=False,method='min').loc[city]),
            'n':len(peer),'weak':weak,'percentile':float(percentiles.loc[city,weak]),
            'type':title,'action':action,'monitor':monitor,'level':row['城市层级']}

def entropy_audit(source=None):
    """Reproduce global entropy weights and total index from supplied normalized data."""
    source = source or ROOT/'data/source.xlsx'
    z = pd.read_excel(source,sheet_name='步骤1_标准化数据').set_index(['城市','年份'])
    weights = pd.read_excel(source,sheet_name='步骤3-5_熵值_差异系数_权重').set_index('指标')
    z = z[weights.index].astype(float)
    if not np.isfinite(z).all().all() or (z<0).any().any() or (z.sum()==0).any():
        raise ValueError('标准化表存在无效数值，不能复算。')
    p = z.div(z.sum(axis=0))
    entropy = -(p * np.log(p.where(p>0,1))).sum()/np.log(len(p))
    w = (1-entropy)/(1-entropy).sum()
    supplied_p = pd.read_excel(source,sheet_name='步骤2_比重矩阵P').set_index(['城市','年份'])[weights.index].astype(float)
    if not supplied_p.index.equals(z.index):
        raise ValueError('比重矩阵与标准化表的城市年份顺序不一致。')
    epsilon_entropy=-(supplied_p*np.log(supplied_p.where(supplied_p>0,1))).sum()/np.log(len(p))
    epsilon_w=(1-epsilon_entropy)/(1-epsilon_entropy).sum()
    scores = z.dot(weights['权重W'])
    actual = pd.read_excel(source,sheet_name='综合指数得分').set_index(['城市','年份'])[INDEX]
    aligned = pd.concat([scores.rename('复算'),actual.rename('原表')],axis=1)
    if aligned.isna().any().any():
        raise ValueError('标准化数据与指数表的城市年份键不一致。')
    return weights.reset_index(), {'n':len(z),'weight_sum':float(weights['权重W'].sum()),
        'weight_max_error':float((w-weights['权重W']).abs().max()),
        'source_p_weight_max_error':float((epsilon_w-weights['权重W']).abs().max()),
        'p_max_error':float((supplied_p-p).abs().max().max()),
        'score_max_error':float((aligned['复算']-aligned['原表']).abs().max())}

def workbook_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def report_markdown(d, sim=None, group=None):
    text = f'''# {d['city']}创新生态决策简报

数据年份：{d['year']}；参照范围：当年{d['n']}个样本城市。数据为用户提供的插值后历史面板，不代表实时监测。

## 城市诊断
综合指数 {d['index']:.6f}，排名 {d['rank']}/{d['n']}。城市层级：{d['level']}。
相对短板：{d['weak']}，当年样本百分位 {d['percentile']:.1f}%。短板根据四维百分位最低项确定，不是绝对不合格判断。

## 待论证行动
方向：{d['type']}。
{d['action']}
后续建议观察：{d['monitor']}。
以上为透明规则生成的研究建议，未进行实施效果验证。
'''
    if sim:
        text += f'''\n## 条件情景对比
系数组：{group}；指数条件变化 {sim['delta']:.6f}，系数不确定性近似区间 [{sim['lower']:.6f}, {sim['upper']:.6f}]。
使用公式 ΔInn=β×[ln(1+G×(1+r))−ln(1+G)]×1(year≥2015)。不确定性只包括系数标准误，非完整预测区间。
基金口径由操作者确认后使用；未重新估计原论文模型，不构成实际政策效果承诺。
'''
    return text+'\n## 证据边界\n原文中介样本为492、基准615；市场化分组缺失；Excel熵值法与正文CRITIC表述待统一；基金累计/年度口径待核对。研发、数字金融已纳入总指数，机制分析存在构造性相关风险。\n'
