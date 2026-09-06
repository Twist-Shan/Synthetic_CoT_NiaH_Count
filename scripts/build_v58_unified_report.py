"""Build the uniform-budget body-mechanism report; never mix historical panels."""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_v58_synthetic_report import geometry_projection_widget
from v58_alignment_core import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'work/v58_final'
ALIGN = DATA/'analysis/v58_alignment_supplement_20260905'
LEGACY = DATA/'analysis/v58_unified_legacy_20260905'
EXTRA = DATA/'analysis/v58_unified_additional_20260905'
ASSETS = ROOT/'reports/assets/v58_unified_20260905'
MODES = ['nonthinking', 'thinking']
COLORS = {'nonthinking': '#ce7241', 'thinking': '#267cb0'}


def read(path):
    return pd.read_csv(path)


def pct(x):
    return f'{100*x:.1f}%'


def table(frame, columns=None, digits=3):
    if columns is not None:
        frame = frame[columns]
    return '<div class="table-scroll">'+frame.to_html(index=False, border=0, na_rep='—', float_format=lambda x: f'{x:.{digits}f}')+'</div>'


def fig(fig, filename, caption):
    path = ASSETS/filename
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return '<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+caption+'</figcaption></figure>'


def conclusion(text):
    return f'<p class="conclusion"><b>当前结论。</b>{text}</p>'


def contrast(left, right, metric, keys=('key',), scale=1):
    a = left.groupby(list(keys)).agg(value=(metric, 'mean'), block=('block', 'first')).reset_index()
    b = right.groupby(list(keys))[metric].mean().reset_index(name='control')
    merged = a.merge(b, on=list(keys), validate='one_to_one')
    assert len(merged) == len(a) == len(b)
    result = paired_bootstrap((merged.value-merged.control).to_numpy()*scale, merged.block.to_numpy())
    return f"{result['effect']:.2f} [{result['ci_low']:.2f}, {result['ci_high']:.2f}]"


def behavior_and_ablation(ablation):
    clean = ablation.loc[ablation.arm.eq('clean')]
    bycount = clean.groupby(['mode', 'count']).ar_accuracy.agg(['mean', 'size']).reset_index()
    fig1, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout='constrained')
    for mode in MODES:
        frame = bycount.loc[bycount['mode'].eq(mode)]
        axes[0].plot(frame['count'], frame['mean']*100, 'o-', color=COLORS[mode], label=mode)
        sub = ablation.loc[ablation['mode'].eq(mode)]
        for arm, style in [('selected', '-'), ('random', '--')]:
            d = sub.loc[sub.arm.eq(arm)].groupby('top_k').ar_accuracy.mean()
            axes[1].plot(d.index, d.values*100, 'o'+style, color=COLORS[mode], label=mode+' '+('selected' if arm=='selected' else 'matched controls'))
    axes[0].set(xlabel='True count', ylabel='Free-running answer accuracy (%)', xticks=range(1, 11), ylim=(-3, 103))
    axes[1].set(xlabel='Ablated heads K', ylabel='Free-running answer accuracy (%)', xticks=[1, 2, 4], ylim=(-3, 103))
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    image = fig(fig1, 'behavior_ablation.png', '图1｜左：每个数字10个相同confirmation输入的答案准确率；横轴为真实count，纵轴为自由生成正确率。右：同一100个输入的K剂量曲线，实线为按discovery选择的检索头，虚线为同层、等头数、不重叠对照的均值。K=1/2各3个不同对照，K=4仅1个可行补集；虚线不代表随机分布。无误差条，配对区间见正文。')
    summary = clean.groupby('mode')[['ar_accuracy', 'trace_exact', 'trace_marker_count_accuracy']].mean().reset_index()
    effects = []
    for mode in MODES:
        f = ablation.loc[ablation['mode'].eq(mode)]
        for k in [1, 2, 4]:
            selected = f.loc[f.arm.eq('selected') & f.top_k.eq(k)]
            controls = f.loc[f.arm.eq('random') & f.top_k.eq(k)]
            effects.append({'mode': mode, 'K': k, 'selected_answer': selected.ar_accuracy.mean(),
                'control_answer': controls.ar_accuracy.mean(),
                'selected_answered': selected.ar_answered.mean(), 'control_answered': controls.ar_answered.mean(),
                'selected_minus_control_pp_95CI': contrast(selected, controls, 'ar_accuracy', scale=100),
                'selected_trace_exact': selected.trace_exact.mean(), 'control_trace_exact': controls.trace_exact.mean()})
    return image, summary, pd.DataFrame(effects), bycount


def plot_dynamics():
    data = pd.concat([read(LEGACY/m/'dynamics_attention_summary.csv') for m in MODES])
    behavior = pd.concat([read(LEGACY/m/'dynamics_behavior_trials.csv') for m in MODES])
    fig1, axes = plt.subplots(3, 2, figsize=(13, 13), layout='constrained')
    for row, (mode, role) in enumerate([('nonthinking', 'broad'), ('thinking', 'targeted'), ('thinking', 'successor')]):
        f = data.loc[data['mode'].eq(mode)].copy()
        f['physical_head'] = ['L%dH%d'%(l,h) for l,h in zip(f.layer,f['head'])]
        matrix = f.pivot(index='physical_head', columns='step', values=role)
        order = [f'L{l}H{h}' for l in range(1,5) for h in range(8)]
        matrix = matrix.reindex(order)
        matrix = matrix.div(matrix.sum(axis=0), axis=1).fillna(0)
        for col in range(2):
            steps = matrix.columns.to_numpy(); mask = steps>0 if col else np.ones(len(steps), bool)
            x = steps[mask].astype(float); z = matrix.iloc[:, mask].to_numpy()
            im = axes[row,col].pcolormesh(x, np.arange(32), z, shading='nearest', cmap='magma', vmin=0, vmax=max(.16, matrix.to_numpy().max()))
            if col:
                axes[row,col].set_xscale('log')
            axes[row,col].set(xlabel='Optimizer step (log scale)' if col else 'Optimizer step',
                yticks=np.arange(32), yticklabels=order, title=f'{mode}: {role} role share')
            axes[row,col].tick_params(axis='y', labelsize=6)
            axes[row,col].axvline(1500, color='#67bdce', lw=1, ls='--')
            axes[row,col].invert_yaxis()
            fig1.colorbar(im, ax=axes[row,col], label='Head score / sum over 32 heads')
    image = fig(fig1, 'role_heatmaps.png', '图4｜固定100个confirmation输入，按物理层/头排序的分化热图。上：Non-thinking answer-side broad score；中：Thinking trace-query targeted mass；下：Thinking marker→preceding-separator attention。横轴分别为线性steps与log steps，颜色为该头在32个头同一指标总和中的份额。每100步一个checkpoint；log轴省略step 0；青色虚线为1500步loss-scope切换。两列使用相同数据，log变换只改变显示间距，不能证明突变。')
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for mode in MODES:
        b = behavior.loc[behavior['mode'].eq(mode)].groupby('step').ar_accuracy.mean()
        axes2[0].plot(b.index, 100*b, 'o-', label=mode, color=COLORS[mode])
        f = data.loc[data['mode'].eq(mode)]
        sites = json.loads((ALIGN/mode/'frozen_sites.json').read_text())
        role = 'broad' if mode=='nonthinking' else 'targeted'
        heads = [tuple(x[:2]) for x in sites['ranking'][role][:4]]
        selected = f.loc[[tuple(x) in heads for x in f[['layer','head']].to_numpy()]]
        axes2[1].plot(selected.groupby('step')[role].mean(), label=mode+' fixed Top-4', color=COLORS[mode])
        share = selected.groupby('step')[role].sum()/f.groupby('step')[role].sum()
        axes2[2].plot(share, label=mode, color=COLORS[mode])
    axes2[0].set(xlabel='Optimizer step', ylabel='Free-running accuracy (%)', ylim=(0,103))
    axes2[1].set(xlabel='Optimizer step', ylabel='Mean fixed-bank role score')
    axes2[2].set(xlabel='Optimizer step', ylabel='Fixed Top-4 role share', ylim=(0,1)); axes2[2].axhline(4/32,ls=':',color='gray')
    for ax in axes2:
        ax.legend(fontsize=8); ax.grid(alpha=.2); ax.axvline(1500, ls='--', color='gray', lw=.8)
    image += fig(fig2, 'dynamics_behavior_bank.png', '图5｜横轴均为steps。左：预先固定11个checkpoint、每mode每点相同100个输入的自由生成准确率；中：最终200个discovery输入冻结的Top-4 bank的平均原始role score；右：该bank占32个heads总分的比例，点线为4/32参考份额。Broad score与targeted mass定义不同，不比较两者绝对数值高低；不在每个checkpoint重新选头。')
    fig3, axes3 = plt.subplots(1, 2, figsize=(12,4), layout='constrained')
    for mode in MODES:
        f=read(LEGACY/mode/'dynamics_causal_trials.csv')
        for metric,style in [('ar_accuracy','-'),('trace_exact',':')]:
            if mode=='nonthinking' and metric=='trace_exact':
                continue
            s=f.loc[f.condition.eq('selected')].groupby('step')[metric].mean()
            c=f.loc[f.condition.eq('control')].groupby('step')[metric].mean()
            axes3[0].plot(s.index,100*(c-s),'o'+style,label=mode+' '+metric,color=COLORS[mode])
    transport=[]
    for folder in sorted((LEGACY/'thinking').glob('transport_step_*')):
        frame=read(folder/'transport_trials.csv'); frame['step']=int(folder.name.rsplit('_',1)[1]); transport.append(frame)
    transport=pd.concat(transport)
    for condition,style in [('value_selected','-'),('value_control','--')]:
        f=transport.loc[transport.condition.eq(condition)&transport.top_k.eq(2)].groupby('step').restoration.mean()
        axes3[1].plot(f.index,f,'o'+style,label=condition)
    axes3[0].set(xlabel='Optimizer step',ylabel='Control accuracy minus selected accuracy (pp)')
    axes3[1].set(xlabel='Optimizer step',ylabel='Patched minus damaged marker logit margin')
    for ax in axes3:
        ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=8)
    image+=fig(fig3,'dynamics_causal_transport.png','图5b｜同一11个checkpoint、每条件相同100输入。左：Top-2选中头相对三个等层对照的额外损伤，纵轴为control正确率减selected正确率（百分点）；实线为最终答案，点线为Thinking trace exact，正值表示选中头损伤更大。右：Thinking对应source的V恢复后相对损坏baseline的marker logit margin增益，实线选中Top-2，虚线同层对照均值。没有归一化小分母，也没有在训练中重新选头。')
    return image


def continuation_section(registry):
    root = LEGACY/'continuation'
    plan = json.loads((root/'plan.json').read_text())
    selected = json.loads((root/'selected_layers.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert plan['discovery_prompts']==20 and plan['confirmation_prompts']==10
    pairs = read(root/'frozen_pairs.csv')
    assert set(pairs.loc[pairs.split.eq('confirmation'), 'prompt_sha256']) == set(registry.loc[registry.split.eq('confirmation') & registry['count'].eq(10),'key'])
    rows, selection_rows = [], []
    for scope in ['item_end_w1', 'item_span_w2']:
        info = selected['scopes'][scope]
        for row in info['layer_summaries']:
            selection_rows.append(dict(scope=scope, selected_layer=info['selected_layer'], **row))
        if info['selected_layer'] is None:
            continue
        trials = read(root/scope/'rollout_trials.csv')
        for metric in ['donor_marker_adoption','donor_continuation_adoption','donor_prefix_h2','donor_prefix_h3','donor_prefix_h4']:
            f = trials.loc[trials[metric].notna()].copy()
            if metric == 'donor_marker_adoption':
                f = f.loc[f.successor_identity_distinct]
            f['key'] = f.pair_id
            f['block'] = f.prompt_sha256.map(registry.set_index('key').block)
            d = f.loc[f.condition.eq('full_donor_patch')]
            c = f.loc[f.condition.eq('self_patch')]
            o = f.loc[f.condition.str.startswith('full_norm_orthogonal')]
            def promptmean(x):
                return x.groupby('prompt_sha256')[metric].mean().mean()
            rows.append({'scope': scope, 'layer': info['selected_layer'], 'metric': metric,
                'eligible_pairs': len(d), 'eligible_prompts': d.prompt_sha256.nunique(),
                'self': promptmean(c), 'donor': promptmean(d), 'orthogonal': promptmean(o),
                'donor_minus_orth_pp_95CI': contrast(d,o,metric,scale=100) if len(d) else 'not identifiable'})
    return pd.DataFrame(rows), pd.DataFrame(selection_rows), manifest


def continuation_plot(frame, selection):
    figure, axes = plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for scope,f in selection.groupby('scope'):
        axes[0].plot(f.layer,f.median_prompt_mean,'o-',label=scope)
        axes[0].axvline(f.selected_layer.iloc[0],color='#888',ls=':',lw=.8)
    axes[0].set(xlabel='Patched post-block layer',ylabel='Discovery donor/receiver log-odds shift',xticks=[1,2,3,4])
    for scope,style in [('item_span_w2','-'),('item_end_w1',':')]:
        f=frame.loc[frame.scope.eq(scope)]
        x=np.arange(len(f))
        axes[1].plot(x,100*f.donor,'o'+style,label=scope+' donor')
        axes[1].plot(x,100*f.orthogonal,'s'+style,label=scope+' orth.')
        values=np.array([[float(t) for t in re.findall(r'-?\d+(?:\.\d+)?',s)] for s in f.donor_minus_orth_pp_95CI])
        offset=.05 if scope=='item_span_w2' else -.05
        axes[2].errorbar(x+offset,values[:,0],yerr=[np.maximum(0,values[:,0]-values[:,1]),np.maximum(0,values[:,2]-values[:,0])],fmt='o',capsize=3,label=scope)
    for ax in axes[1:]:
        ax.set(xticks=np.arange(5),xticklabels=['Next','q<=4','h=2','h=3','h=4'],xlabel='Eligible continuation endpoint')
    axes[1].set_ylabel('Donor-prefix adoption (%)');axes[2].set_ylabel('Donor minus norm control (pp)')
    axes[2].axhline(0,color='gray',lw=.8)
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    return fig(figure,'progress_continuation.png','图3｜左：20个discovery prompts的可识别pair选择层，横轴post-block层，纵轴donor/receiver logodds变化的prompt平均后中位数；竖虚线为冻结选层L1。中：统一10-prompt confirmation子集中的donor与等范数对照前缀adoption；右：两者配对百分点差及prompt-clustered 95%区间。横轴五个指标依次含26、37、30、26、20个可识别pairs，均来自8个prompts；各指标分母不同，连线仅辅助阅读。')


def build(output):
    output = output.resolve()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for root in [ALIGN, LEGACY, EXTRA]:
        assert json.loads((root/'manifest.json').read_text())['status']=='complete', root
    assert json.loads((LEGACY/'unified_sample_audit.json').read_text())['status']=='passed'
    registry = read(ALIGN/'input_registry.csv')
    assert len(registry)==300 and registry.key.nunique()==300
    expected = set(registry.loc[registry.split.eq('confirmation'),'key'])
    trials = pd.concat([read(ALIGN/m/'trials.csv') for m in MODES])
    ablation = pd.concat([read(ALIGN/m/'ablation.csv') for m in MODES])
    for (_,_,_,_), f in ablation.groupby(['mode','arm','top_k','repeat']):
        assert set(f.key)==expected and len(f)==100
    for mode in MODES:
        for family in ['source','answer_source']:
            for _,f in trials.loc[trials['mode'].eq(mode)&trials.family.eq(family)].groupby(['arm','layer'],dropna=False):
                assert set(f.key)==expected and len(f)==100
    image1, behavior, abl_effects, bycount = behavior_and_ablation(ablation)
    clean_geometry = pd.concat([read(LEGACY/m/'geometry/clean_layer_metrics.csv') for m in MODES])
    selections = pd.concat([read(LEGACY/m/'geometry/clean_selections.csv') for m in MODES])
    selected_geometry = selections.loc[selections.selector.eq('ncc_balanced_accuracy')]
    cloud = pd.concat([read(LEGACY/m/'geometry/projection_cloud.csv') for m in MODES])
    cloud['sample'] = cloud.prompt_sha256.map({k:i for i,k in enumerate(registry.key)})
    widget = geometry_projection_widget(cloud).replace('<option value="0">L0 · embedding output</option>','')
    probe = pd.concat([read(LEGACY/m/'geometry/frozen_probe_trials.csv') for m in MODES])
    # Balanced accuracy weights the ten labels equally, not the more numerous low-k occurrences.
    probe_summary = probe.groupby(['mode','endpoint','layer','condition','top_k','repeat','occurrence']).ncc_correct.mean().groupby(['mode','endpoint','layer','condition','top_k','repeat']).mean().reset_index()
    probe_selected=[]
    for m in MODES:
        depths=json.loads((LEGACY/m/'geometry/frozen_depths.json').read_text())
        for ep,l in depths.items():
            f=probe_summary.loc[probe_summary.endpoint.eq(ep)&probe_summary.layer.eq(l)]
            probe_selected.append(f)
    factorial=read(LEGACY/'thinking/factorial_trials.csv')
    transport=read(LEGACY/'thinking/transport_trials.csv')
    transport_summary=transport.groupby(['condition','top_k','repeat']).agg(prompts=('key','nunique'),clean_margin=('clean_margin','mean'),damaged_margin=('corrupt_margin','mean'),patched_margin=('margin','mean'),restoration=('restoration','mean')).reset_index()
    source=trials.loc[trials.family.eq('source')].groupby(['mode','arm','layer'],dropna=False)[['accuracy','margin','expected_abs_error','running_centroid_distance_l1','running_centroid_distance_l2']].mean().reset_index()
    answer_source=trials.loc[trials.family.eq('answer_source')].groupby(['mode','arm'])[['accuracy','margin','expected_abs_error','count_probability_mass']].mean().reset_index()
    answer=trials.loc[trials.family.eq('answer')].copy()
    answer['donor_adoption']=np.where(answer.offset.notna(), (answer.predicted_count==answer['count']+answer.offset).astype(float), np.nan)
    answer_summary=answer.groupby(['mode','arm','layer'])[['accuracy','donor_adoption','margin']].mean().reset_index()
    native_edges = np.array([(int(n),int(n+o)) in {(1,2),(2,1),(5,6),(6,5)} if pd.notna(o) else False for n,o in zip(answer['count'],answer.offset)])
    donor_native = answer.loc[answer.arm.eq('adjacent_donor') & native_edges]
    native_summary=donor_native.groupby(['mode','layer']).agg(pairs=('key','size'),donor_adoption=('donor_adoption','mean'),receiver_accuracy=('accuracy','mean')).reset_index()
    bridge=trials.loc[trials.family.eq('terminal_bridge')].groupby(['arm','scope'],dropna=False)[['accuracy','margin']].mean().reset_index()
    relay=trials.loc[trials.family.eq('terminal_relay')].copy()
    relay_summary=relay.groupby(['arm','reset'])[['accuracy','margin']].mean().reset_index()
    serial=trials.loc[trials.family.eq('serial')].groupby(['source_layer','source_restored','retrieval','late'])[['accuracy','margin','retrieval_centroid_distance_l1','answer_centroid_distance_l4']].mean().reset_index()
    source_next=read(EXTRA/'thinking/source_next_trials.csv')
    source_next_summary=source_next.groupby(['arm','ordinary_control']).agg(prompts=('key','nunique'),nonempty=('blank_tokens',lambda x:int((x>0).sum())),marker_accuracy=('next_marker_correct','mean'),margin=('marker_margin','mean')).reset_index()
    vector=pd.concat([read(EXTRA/m/'count_vector_trials.csv') for m in MODES])
    vector_summary=vector.groupby(['mode','arm','layer'])[['donor_count_adoption','directed_expected_shift']].mean().reset_index()
    progress, progress_selection, progress_manifest=continuation_section(registry)
    progress_image=continuation_plot(progress,progress_selection)
    images_dynamics=plot_dynamics()
    gd=pd.concat([read(EXTRA/m/'geometry_dynamics_trials.csv') for m in MODES])
    gds=gd.groupby(['endpoint','step','occurrence']).ncc_correct.mean().groupby(['endpoint','step']).mean().reset_index()
    figg, axg=plt.subplots(1,1,figsize=(9,4),layout='constrained')
    for ep,f in gds.groupby('endpoint'):
        axg.plot(f.step,100*f.ncc_correct,'o-',label=ep)
    axg.set(xlabel='Optimizer step',ylabel='Balanced NCC accuracy (%)',ylim=(0,103));axg.legend(fontsize=8);axg.grid(alpha=.2)
    geometry_dynamic_image=fig(figg,'geometry_dynamics.png','图6｜横轴为预先固定的11个训练checkpoint；纵轴为十个标签等权的confirmation NCC准确率。每个endpoint的物理层由最终discovery CV确定后固定，每个checkpoint单独用相同200个discovery输入拟合标准化、PCA16与centroids，再评相同100个confirmation输入。Running状态包含每条输入的全部occurrences。该图包含重新拟合的probe，不代表固定decoder跨checkpoint的迁移。')
    nt=behavior.loc[behavior['mode'].eq('nonthinking'),'ar_accuracy'].iloc[0]; th=behavior.loc[behavior['mode'].eq('thinking'),'ar_accuracy'].iloc[0]
    coverage=pd.DataFrame([
        ['Behavior / Top-K','200 / 100','10/count; both modes','K=1,2: 3 controls; K=4: 1','完成'],
        ['Four-endpoint geometry / post-Top-K NCC','200 / 100','1100 / 550 running states; 200 / 100 answer states','same input + occurrence keys','完成'],
        ['Source formation / restoration','200 / 100','100 per mode per condition','embedding + L1–L4, ordinary control','完成'],
        ['Answer state / count directions','200 / 100','180 directed pairs/depth; Native slice 40','self + context / 3 norm controls','完成'],
        ['NT retrieval write / serial factorial','200 / 100','100 per combination','2 source depths × 2 × 3 × 3','完成；positive mediation待检验'],
        ['Thinking source-next / answer source','200 / 100','100 per condition','empty-history support reported separately','完成'],
        ['Thinking terminal bridge / relay','200 / 100','100 / 180 pairs per condition','same damaged baseline / downstream reset','完成；bridge范围受限'],
        ['Read → separate carrier → commit','200 / 100','90 inputs with N≥2','two-token item lacks separate post-marker commit','架构不适用；不能计为复现'],
        ['Progress free continuation','20 / 10 (count-10 subset)','40 discovery / 60 confirmation pairs/scope','8 conditions; 2 scopes','完成'],
        ['Training dynamics','fixed 200 / 100','same inputs at every checkpoint','100-step attention; 11 behavior milestones','完成；single training seed'],
    ],columns=['Experiment','Discovery / confirmation prompts','Realized unit','Control / restriction','Status'])
    cfg=json.loads((DATA/'config.json').read_text())
    train=read(DATA/'tables/train_metrics.csv')
    sampling=read(DATA/'tables/training_sampling_distribution.csv')
    sampling=sampling.loc[sampling.dimension.eq('accepted_counts')].copy()
    sampling['fraction']=sampling.examples/sampling.total_training_examples
    setup=pd.DataFrame([
        ['Data','Shakespeare chars；target set=3 chars；100 sets；query first；256-char context；count1–10'],
        ['Sampler','max-entropy set×count；训练窗口重新随机排列字符；集合顺序打乱；corpus split 80/10/10'],
        ['Models','两份独立初始化、独立优化的12,658,176参数模型；4层×8heads；d512；MLP2048；RoPE base10000；384 positions'],
        ['Training','每mode 10000 steps×batch128=1.28M examples；AdamW lr3e-4，warmup500，wd.01，clip1，BF16；single seed1234'],
        ['Trace','<Think> (<Sep> marker)^N </Think> <Ans> <N> <EOS>；无显式running index；NT只有<Ans> <N> <EOS>'],
        ['Loss steps1–1500','所有非padding位置的teacher-forced next-token cross-entropy'],
        ['Loss steps1501–10000','task-output；分区归一化 count/marker/structure系数8/8/16；T权重份额25/25/50%，NT33.3/66.7%'],
        ['Readout','count1–10为atomic单token且独立输出行；其余词表输入/输出embedding tied；两mode相同；无TTT或联合mode训练'],
        ['Checkpoint','每100steps保存fp16科学snapshot；每500steps保存optimizer恢复状态；本轮不重训、不改trace'],
    ],columns=['Setting','Value'])
    # Training loss has a changing objective; never concatenate as one homogeneous risk.
    trainfig, tax=plt.subplots(1,1,figsize=(9,3.6),layout='constrained')
    losscol=next((c for c in ['train_total_loss','loss','train_loss'] if c in train),None)
    if losscol:
        for mode,f in train.groupby('mode'):
            tax.plot(f.step,f[losscol],label=mode)
        tax.axvline(1500,color='gray',ls='--'); tax.set(xlabel='Optimizer step',ylabel='Logged training loss');tax.legend();tax.grid(alpha=.2)
    loss_image=fig(trainfig,'training_loss.png','图0｜原始训练日志。横轴为optimizer step，纵轴为当时训练目标下记录的loss。1500步后loss作用域与分区归一化发生变化，因此切换前后绝对loss不应视作同一定义的连续风险；该训练日志不属于confirmation实验样本量。')
    body=f'''<h1>NiaH Synthetic v58：统一规模的机制复验</h1>
<p class="subtitle">2026-09-05 · 独立训练的 Non-thinking / separator Thinking · step 10000 · 原始模型与trace保持不变</p>
<div class="summary"><b>结果概览。</b>统一100个confirmation输入上，Thinking自由生成准确率{pct(th)}，Non-thinking {pct(nt)}。
检索、表征、干预和continuation分别评估。本文区分“实验已完成”“协议对齐”“机制得到正面复现”；完成一个干预不自动意味着其机制成立。</div>
<p>旧的不同规模结果保存在<a href="NiaH_Synthetic_report_pre_alignment_20260905.html">历史报告</a>，不参与本文数值汇总。先前已评过该来源池的行为，当前面板属于预注册的机制扩展，不能称为全新未见行为测试。</p>
<h2>1. 模型、训练与统一样本规则</h2>{table(setup)}{loss_image}
<p>训练count占比范围为{sampling.fraction.min()*100:.3f}%–{sampling.fraction.max()*100:.3f}%。新评估冻结200个discovery输入（20/count）与100个confirmation输入（10/count），双方使用同一输入、相同gold count和occurrence标签。
每个confirmation block含10个不同输入，各count一个；10个block是预先规定的统计分组，不代表10个独立模型seed，也不等价于大模型的同一生成seed跨count系列。
旧head selection及先前progress/continuation的canonical输入排除后按hash选择，未按正确率、NCC或干预结果筛选。</p>
<p><b>统计定义。</b>准确率以输入为分母；running NCC先按标签1–10分别计算再等权平均。配对效应在同一输入或donor pair上相减、block内平均，再对10个block做10000次bootstrap，给出未作多重比较校正的95%区间。
Donor pairs、层、heads、重复对照和token状态均非新增独立输入。Count-10 continuation只有10个confirmation prompts，重复字符导致可识别子集进一步减少，区间精度有限。</p>
<p>表格中的accuracy/adoption/NCC数值默认是0–1比例，pp表示百分点；margin为自然对数logit差，距离和方向指标按各节定义。所有下表均来自统一面板，历史训练日志单独标注。</p>
<p><b>说明性示例。</b>同一题有6个needle：NT取正文第1–6个needle位置作为running状态，Thinking取trace的第1–6个marker末端；两边都有6条状态记录，但独立输入数仍是1。</p>
{conclusion('旧的正文级实验已按同一面板重算。实验类型决定派生的token/pair数量，不能强行把这些数量当成相等的独立样本量。训练预算匹配，但本次只有一个训练seed。')}
{table(coverage)}
<h2>2. 自由生成行为与检索头必要性</h2>
<p><b>目的与定义。</b>检验两mode行为差异及检索头对答案、trace内容、trace长度的分别贡献。Greedy生成直到EOS或预定上限，缺失答案计错。
Broad score为M·exp(H)/N：M是answer query分配到全部needle的attention mass，H是needle内归一化attention的熵；targeted score为每个trace query指向对应needle的attention，再先在prompt内平均。
200个discovery输入选择最终Top-K；所有confirmation结果不参与选择。Head编号L1–L4，H0–H7。</p>
<p><b>干预范围与例子。</b>NT仅在&lt;Ans&gt;预测count的位置清零选中heads的pre-O输出；Thinking在trace内每个&lt;Sep&gt;预测marker的位置清零，随后继续自由生成。例如删掉第3次检索头后可产生错误marker，但仍生成正确数量的items。
本轮修正了旧代码同时包含prompt-query分隔符的范围问题；旧数值仅保留作历史记录。对照为预先确定的不同同层补集组合，不声称随机抽样分布。</p>
{image1}{table(behavior)}{table(abl_effects)}
<p><b>Non-thinking对照的输出故障。</b>选中Top-4消融后100/100仍给出数值答案，但只有7题正确；唯一同层补集消融后0/100给出数值答案，K2的两个对照也分别只有3/100和0/100给出数值答案。
因此对照的0%准确率同时包含输出格式失败，不能视作干净的count-specific损伤比较。保留这些对照，不按结果重新抽取。为何该补集导致通用输出故障的具体计算原因未查明。</p>
{conclusion('Thinking答案优势在统一输入上保留。检索头对trace内容和最终count的影响需分开报告；Non-thinking对照包含明显通用输出故障，当前未建立broad bank的干净特异性必要性，不能声称两个mode均完成了相同强度的检索消融复现。')}
<details><summary>每count结果与Thinking factorial（同一100输入）</summary>{table(bycount)}{table(factorial.groupby('arm')[['ar_accuracy','trace_exact','trace_marker_count_accuracy']].mean().reset_index())}</details>
<p><b>Role specialization的因果对照。</b>同一100输入上，targeted Top-2使trace exact从87%降至62%，count准确率95%→91%；discovery选中的successor L2H3在marker位置消融后，count准确率降至45%，同层三个对照为79%、94%、83%。
在answer query消融Thinking broad Top-2后仍为95%，与targeted联合消融后91%。这些结果支持已测试位置上的角色分工；不能把L2H3的作用直接解释为纯数值加一，也不能外推到所有broad heads。</p>
{conclusion('Targeted内容检索与支持正确trace长度/答案的successor作用可在同一面板上区分，解释了targeted消融对trace内容的伤害大于对count的伤害。具体计算方式仍需进一步定位。')}
<h2>3. Geometry：running index 与 final count</h2>
<p><b>目的。</b>检验已完成计数进度与最终count能否从状态读出、类内变化是否较小，并检验Top-K消融是否改变这些表征。</p>
<p><b>设置与计算。</b>四个endpoint是NT正文needle末端、Thinking trace marker末端及各自&lt;Ans&gt;位置。Running共1100个discovery状态/550个confirmation状态；final共200/100。
Geometry、NCC和attention dynamics使用固定gold trace的teacher-forced前向；自由生成行为及progress rollout另行评估。因此早期NCC可利用给定的正确trace，不表示早期模型已经能自己生成该trace。
StandardScaler与whitened PCA≤16只在discovery拟合；NCC在该空间中选择最近类中心。5-fold grouped discovery CV选择层；正文报告NCC-selector并另列共同decoder选层结果。
二维/三维图使用discovery拟合的非whitened PCA3投影；各mode/endpoint空间独立，不比较跨图绝对距离。图中看起来紧凑不能替代量化指标。</p>
<p><b>说明性示例。</b>第3个trace marker状态的running标签是3，最终有7个needle时&lt;Ans&gt;状态的final标签是7。若最近discovery类中心为3，前者NCC预测正确。</p>
{table(selected_geometry,['endpoint','selected_layer','discovery_value','confirmation_value','common_decoder_selected_layer'])}{widget}
<p class="caption">图2｜左右为Non-thinking/Thinking，上下为running index/final count；2D与3D可分别选L1–L4，3D支持拖拽旋转。坐标为PC1/PC2/PC3，轴上注明discovery方差比例；散点为confirmation状态，颜色为k或N，红线连接类均值供观察。不同层不共用PCA坐标系。</p>
{conclusion('NCC说明状态中的标签信息可读出。Thinking固定两token/item导致答案位置与N线性对应，因此高final NCC不能单独证明内容无关的内部计数器或语义压缩。')}
<h3>Top-K之后的冻结clean probe</h3>
<p>以下在clean discovery上拟合probe后冻结，评估相同confirmation状态的query-local消融，不在损坏数据上重新训练probe。报告共同decoder-selected物理层；完整all-layer结果可复查。
Thinking targeted heads位于末层时，较早层状态以及同层更晚固定token状态在该计算图上均不能受其query-local mask影响；NCC不变属于时间顺序限制。</p>
{table(pd.concat(probe_selected))}
{conclusion('消融后NCC已经补齐；若干预位置在被读出状态之后，NCC不变不能作为该状态不参与任务的证据。')}
<details><summary>完整all-layer geometry指标</summary>{table(clean_geometry)}</details>
<h2>4. Source formation、value transport 与同一forward的中介检验</h2>
<p><b>目的与例子。</b>把某个needle字符替换为等长度普通字符，在受损输入上恢复clean source状态，检验正文source信息是否影响后续答案。例：把第3个目标a改成空格，再只恢复该位置的clean embedding或L1–L4状态；普通位置恢复是等token预算对照。</p>
<p>两mode每条件100个输入，Thinking此处使用固定gold trace。源状态保真度为到discovery正确running类中心的Euclidean距离，除以该类discovery RMS半径后在prompt内平均；该raw-space距离与前节PCA-NCC不同。
每个synthetic needle只有1 token，whole span与endpoint相同。Embedding恢复使输入嵌入完全回到clean，是输入身份恢复上界，不能称为学得running state的中介证据。</p>
{table(source)}
{conclusion('NT正文needle损坏后准确率21%→13%；embedding恢复回到21%，L1–L4恢复仅13%–14%，未复现大模型中较强的post-block source-state恢复。Thinking在gold trace固定时对正文source损坏不敏感，说明该条件下的答案读出可利用已有trace。')}
<h3>Targeted value transport（100输入，每题一个固定occurrence）</h3>
<p>固定k=max(1,floor(N/2))，将对应needle身份替换为同目标集合中的另一字符；在同一损坏baseline上，恢复选中head在该source的clean V，或恢复检索query的clean residual。
指标为正确marker与替换marker的logit margin；restoration=patched−damaged。归一化恢复仅在clean−damaged正且可识别时计算，主表保留原始margin，避免小分母夸大。</p>{table(transport_summary)}
{conclusion('Thinking Top-2 value patch将marker margin由−3.783恢复至1.190，三个同层对照平均约−2.769，支持选中heads传输对应marker内容。该局部恢复不等价于整段trace或最终count的充分性；末层完整residual恢复为直接readout上界。')}
<h3>NT source → retrieval write → late answer：factorial</h3>
<p>在同一个损坏forward中交叉source恢复、retrieval write方向删除和晚层answer方向删除。方向来自discovery类中心的前三个奇异向量；对照来自类内残差、与计数子空间正交，并匹配每次实际删除的范数。
检索write取选中answer-side broad bank的post-O求和。当前ret_layer=L1，因此source L1恢复发生在该层retrieval write之后：只有embedding→L1→L4满足所列顺序，source-L1条件不能当作同层write的上游中介。</p>
{table(trials.loc[trials.family.eq('retrieval_natural')].groupby(['mode','arm'])[['accuracy','margin','expected_abs_error']].mean().reset_index())}
<details><summary>全部36个组合，每组合100输入</summary>{table(serial)}</details>
{conclusion('已补同一baseline下的联合干预与中间状态读数。当前结果仍不足以支持NT特异的完整source→retrieval→late→answer串行中介；晚层干预不能回改较早层readout的结构性零效应不构成额外机制发现。')}
<h2>5. 答案读出：状态充分性、方向必要性及来源</h2>
<p><b>目的与例子。</b>保持receiver输入不变，把donor的&lt;Ans&gt; residual写入receiver同一语义位置，观察原题答案是否转为donor count。例如receiver N=5、donor N=6，预测6记作donor adoption。两mode均评相同180个±1有向pairs/层；下表另列Native式1↔2、5↔6的40-pair子集，未按clean正确率筛选。</p>
<p>Self验证logits复现，same-count不同context donor检验内容替换效应；全状态patch保留context/position信息，无法隔离纯数值运算。
Rank-3 count方向必要性使用h−UUᵀ(h−μ)，与真实删除范数相同的orthogonal方向比较。方向充分性额外注入投影后的相邻类中心差，与3个等范数正交方向比较。</p>
{table(native_summary)}<details><summary>全层180-pair结果与方向干预</summary>{table(answer_summary)}{table(vector_summary)}</details>
{conclusion('Thinking在L3/L4 whole-state donor patch后，180/180 pairs输出donor count；NT对应比例较低。Thinking L4计数方向注入的donor-count adoption为45.6%，三个正交对照均0%。这些支持answer-state中的count信息具有因果作用；全状态patch和类中心方向仍保留位置/trace长度混杂，不能直接写成纯±1内部计数。')}
<h3>生成自身trace之后的answer-source必要性</h3>
<p>每个模型先自由生成到&lt;Ans&gt;；本批全部输入到达，随后在embedding和所有post-block层累计清零指定source，保持长度和answer query不变。records为正文needle，trace包括Thinking trace边界/items；ordinary为等token预算非needle位置。NT没有trace，该条件是identity对照。</p>{table(answer_source)}
{conclusion('Thinking在已有自身trace条件下，删除trace比删除正文needles更影响答案，支持trace-to-answer读出。该结果未证明一个必要的universal final aggregator，也不要求作此主张。')}
<h2>6. Terminal bridge、relay 与检索到状态的范围限制</h2>
<p><b>目的与例子。</b>将所有trace items替换成等长度普通字符，再从L2开始累计恢复最后一个item、marker或separator的clean状态。例如trace有5个item，保持总长度不变，只恢复item5；比较同一位置写入相同token预算的普通状态。</p>{table(bridge)}
<p>Relay在180个相邻count pairs上交叉L2 terminal-item donor/self patch与L3后缀/answer-query reset，保留L4作为下游计算。若source确实经该relay影响答案，应观察source×reset的margin交互；先要求自然source效应可测。</p>{table(relay_summary)}
{conclusion('本批terminal bridge恢复较弱，terminal donor对最终答案的影响也小；不能据此建立Native式显著terminal bridge/relay。已存在trace-to-answer来源证据，但terminal局部通路未达到相同证据强度。')}
<p><b>Read→carrier可行性。</b>90个N≥2输入，在最后检索query消融Top-2/4及同层对照，读取item-end all-layer RMS变化。末层query-local mask对同一层其他固定token状态影响为零。
Synthetic item只有&lt;Sep&gt; marker两token，缺少“检索后carrier→更晚commit”的独立位置，无法在保持trace不变的条件下原样复制Native多token carrier闭环。</p>
{conclusion('该项为架构不适用，保留结构性零效应并明确标注，不能计入已复现主体机制。')}
<h2>7. Progress-state patch之后的自由continuation</h2>
<p><b>目的。</b>对齐Native的contextual progress-state实验，检验patch是否让后续生成转向donor的后续marker。采用统一面板的count10子集：20 discovery、10 confirmation；每scope分别选层。
Discovery donor k=6，receiver k=5/7；confirmation donor k=4/6/8、receiver k±1，共60pairs。调整donor的普通filler数量使patch绝对位置对齐，receiver仍256字符；保持十个needle及trace marker顺序。</p>
<p>选层只用discovery distinct-successor logodds变化：forward/backward中位数都为正，选择prompt平均效应中位数达到有效峰值95%的最早层，预先排除末层。随后冻结层与rank3 bases，运行clean、self、full donor、projected donor、projected-norm orthogonal及3个full-norm orthogonal。
从patch位置后完整greedy rollout，不强制下一marker。重复marker造成首身份无法区分时，另用≤4步内最短可区分前缀；各指标保留自身eligible分母，不按生成成功筛选。</p>
<p><b>说明性示例。</b>Receiver未来[a,b,c]，donor未来[a,c,b]；首个a不能区分进度，生成[a,c]才满足最短可区分donor前缀。该证据支持contextual continuation，纯数值±1及最终答案变化属于不同假设。</p>
{progress_image}{table(progress)}<details><summary>Discovery选层全记录</summary>{table(progress_selection)}</details>
{conclusion('统一面板按相同规则选中L1，旧cohort选中的L2保留在历史报告，不沿用旧选层。完整item patch相对等范数对照的首marker adoption提高16.04 pp [4.17,28.96]，两步前缀提高26.04 pp [4.17,52.08]；三步区间跨零，四步区间接触零。本批支持短程contextual continuation，未确认稳定的三/四步效应。')}
<h3>下一次检索依赖哪些source？</h3>
<p>在同一100输入上固定k=max(1,floor(N/2))，在预测第k个marker前累计清零正文records、全部前序items、最近item、除最近之外的items或前半items；普通token数匹配。所有空历史条件仍计入100输入，并单列nonempty支持数。纵向比较使用同一输入，不能把history为空的identity当作机制证据。</p>{table(source_next_summary)}
{conclusion('下一marker准确率clean为96%，清零正文needles后21%，等预算普通token对照95%；清零最近item或全部历史后79%，相应普通对照96%。下一次检索依赖正文source及近期trace状态；已有trace之后的最终答案则表现出不同的来源依赖。')}
<h2>8. Training dynamics：固定面板与固定head bank</h2>
<p><b>目的。</b>检查训练中角色是否逐渐分化，及行为、表征、局部传输与干预效应何时出现。最终discovery冻结head banks；两个独立模型的每个checkpoint使用同一100个confirmation输入。动态热图按固定物理head排序，避免重排制造视觉分化。</p>
<p><b>说明性示例。</b>同一L4H5在step1000与5000上评相同100题。若targeted attention增大，属于该head随训练变化的证据；将每次最大的head排到同一行会混合不同head，本图不这样排序。</p>
{images_dynamics}{geometry_dynamic_image}
<p><b>几何与行为并不同步。</b>固定L2上，Thinking final-count NCC在step400已为100%，当时自由生成答案准确率只有12%；running-index NCC在step200达到72.6%，到step10000为34.2%。
这里的L2是共同decoder选层，与前节按NCC单独选出的running L4（31.4%）不同。早期高NCC与后期行为改善不能合并成“表征随训练持续压缩”的叙述。
固定两token/item使位置/长度携带标签，因此早期NCC的具体来源仍需位置扰动对照；本实验没有隔离这一原因。</p>
{conclusion('Head-bank分化和自由生成能力随训练形成，但当前NCC动态不支持单调增强的representation compression。线性与log横轴呈现同一数据，不把视觉变陡直接解释为相变；1500步loss改变及单训练seed限制formation-step的解释。')}
<details><summary>固定面板动态消融、value transport与raw-centroid压缩统计</summary><p>原始表位于v58_unified_legacy_20260905的dynamics_causal_trials.csv及transport_step_*，均为每条件100输入。Geometry的centroid_rank3_variance与effective_dimension位于v58_unified_additional_20260905；这些是类中心结构统计，不能替代NCC或因果使用证据。</p></details>
<h2>9. 与大模型对齐的结论边界</h2>
<p>参照<a href="../../Realistic_CoT_NiaH_Count/reports/NiaH_Non-thinking_report.html">Non-thinking报告</a>、<a href="../../Realistic_CoT_NiaH_Count/reports/NiaH_Native-Thinking_report.html">Native-thinking报告</a>和<a href="../../Realistic_CoT_NiaH_Count/reports/NiaH_Geometry_Comparison.html">Geometry报告</a>的主体设计，统一了Synthetic内部输入预算、discovery/confirmation隔离、Top-K与同层对照、source restoration、answer状态/方向干预、source necessity、terminal bridge/relay、contextual progress continuation。大模型中的“seed”与此处balanced block不同；两边样本预算相近不能消除任务和模型差异。</p>
<ul><li>较清楚的Synthetic证据：Thinking自由生成优势、targeted检索对trace内容的贡献、successor消融的长度/count效应、trace-to-answer来源依赖、首marker与两步contextual continuation。</li>
<li>未达到相同证据强度：NT post-block source恢复、broad bank的特异性必要性及串行中介；Thinking terminal bridge/relay及三/四步continuation。</li>
<li>保持trace条件下无法精确复制：多token needle的span/endpoint区别、独立retrieved-marker后carrier/commit位置。</li>
<li>外部有效性限制：count只有1–10，256字符，atomic数值，单训练seed；固定两token/item带来位置与trace长度混杂；两mode任务loss分区占比不同；不声称纯内部算术或唯一完整circuit。</li></ul>
{conclusion('当前可以报告主体实验覆盖及具体正/零结果，不能概括为Synthetic已完整复现Non-thinking和Native-thinking全部主体机制。所有未成立或架构不适用项与正结果同等保留。')}
<h2>10. 可复现记录</h2>
<p>输入注册表SHA256：<code>{hashlib.sha256((ALIGN/'input_registry.csv').read_bytes()).hexdigest()}</code>。实验计划、代码、checkpoint hashes、实际覆盖表与逐输入结果均保留。服务器目录：<code>/lambda/nfs/NiaH-Synthetic/runs/{cfg.get('run_name','v58_count1to10_permuted_grammarw16_width512_heads8_steps10000_fullstarts_independent_L256_pool100_seed1234')}/analysis/</code>。</p>
<p>本地结果：<a href="../work/v58_final/analysis/v58_alignment_supplement_20260905/manifest.json">新增主体干预</a> · <a href="../work/v58_final/analysis/v58_unified_legacy_20260905/manifest.json">旧实验统一复算</a> · <a href="../work/v58_final/analysis/v58_unified_additional_20260905/manifest.json">source-next / count方向 / geometry动态</a> · <a href="../work/v58_final/analysis/v58_unified_legacy_20260905/unified_sample_audit.json">实际样本量审计</a> · <a href="../docs/v58_mechanism_alignment_protocol_20260905.md">预定方案与规模规则</a>。</p>
<p class="caption">报告检查：19项测试及实际输入/状态/pair规模审计通过；静态JavaScript语法、5200个geometry投影点和16个endpoint×layer面板覆盖通过检查。浏览器安全策略阻止本地HTML预览，本轮未现场验证3D拖动与选层交互。大型原始状态及逐prompt逐head动态表保留在服务器，报告所需逐输入指标和汇总已下载。</p>'''
    css='''body{margin:0;background:#f2f4f5;color:#23303c;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;padding:40px;background:white}h1{font-size:32px}h2{margin-top:55px;border-top:1px solid #ccd6dc;padding-top:24px}h3{margin-top:30px}.summary,.conclusion{padding:15px 20px;background:#edf5f8;border-left:4px solid #357ea2}.subtitle,.caption,figcaption{color:#596671}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px 11px;text-align:left;border-bottom:1px solid #dde3e7;vertical-align:top}th{background:#f1f5f7}tr:nth-child(even){background:#fafbfc}.table-scroll{overflow:auto;margin:16px 0;max-height:650px}figure{margin:25px 0}figure img{width:100%;height:auto}figcaption{font-size:14px;padding:8px}details{background:#f7f9fa;padding:12px;margin:16px 0}code{overflow-wrap:anywhere;font-size:13px}.geometry-panel{height:760px;width:100%}.geometry-view-title{display:flex;gap:20px;align-items:center;flex-wrap:wrap}select{padding:6px}a{color:#21749d}@media(max-width:700px){main{padding:18px}.geometry-panel{height:650px}}'''
    output.write_text('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NiaH Synthetic — Unified Mechanism Audit</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>',encoding='utf-8')
    for name,frame in [('coverage',coverage),('ablation_effects',abl_effects),('continuation',progress),('geometry_selected',selected_geometry),('behavior',behavior),('answer_native_slice',native_summary)]:
        frame.to_csv(ASSETS/(name+'.csv'),index=False)
    manifest={'schema_version':'v58_uniform_panel_report_v1','report':str(output),'report_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'inputs_sha256':hashlib.sha256((ALIGN/'input_registry.csv').read_bytes()).hexdigest(),'confirmation_prompts':100,
        'count10_confirmation_prompts':10,'historical_results_excluded':True,'progress_validation':progress_manifest.get('validation')}
    manifest['experiment_manifest_sha256']={str(root.relative_to(DATA)) : hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest() for root in [ALIGN,LEGACY,EXTRA]}
    manifest['ui_validation']='static syntax/data coverage and PNG visual checks passed; browser file navigation blocked; interactions not exercised'
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if output.name=='NiaH_Synthetic_report.html':
        output.with_name('NiaH_Synthetic_report_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports/NiaH_Synthetic_report_unified.html')
    build(p.parse_args().output)
