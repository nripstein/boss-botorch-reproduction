"""Reproduce the two report figures. Use scripts/boss-python submission.py."""
import argparse, importlib.util, json, shutil, subprocess, sys, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch

def season():
    spec=importlib.util.spec_from_file_location('season',ROOT/'workflows/seasonality/figure.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def writable(path):
    path=path.resolve()
    for protected in ('results','references','data','verification','src','R','workflows','provenance'):
        p=ROOT/protected
        if path==p or p in path.parents: raise ValueError('Protected submission input directory')
    if path==ROOT or path in ROOT.parents: raise ValueError('Choose a separate output directory')
    return path

def fresh(path):
    path=writable(path)
    if path.exists() and any(path.iterdir()): raise ValueError('Refusing nonempty output; choose a new directory')
    path.mkdir(parents=True,exist_ok=True);return path

def figure2(results,out):
    import io
    from pypdf import PdfReader,PdfWriter,Transformation
    from reportlab.pdfgen import canvas
    subprocess.run(['Rscript','--vanilla',str(ROOT/'workflows/change_point/spline.R'),str(results/'bo_trace.csv'),str(out)],check=True)
    page=PdfReader(ROOT/'references/change_point_layer.pdf').pages[0]
    stream=io.BytesIO();c=canvas.Canvas(stream,pagesize=(576,191*576/361))
    scale=576/361
    for b,(left,bottom,width,height) in zip((20,50),[(131.991,551.346,143.306,142.490),(307.941,551.346,143.306,142.490)]):
        c.saveState();c.scale(scale,scale);c.translate(-102,-518)
        clip=c.beginPath();clip.rect(left,bottom,width,height);c.clipPath(clip,stroke=0,fill=0)
        c.setStrokeColorRGB(1,0,0);c.setLineWidth(.623);c.setLineCap(0);c.setLineJoin(1)
        p=c.beginPath()
        for i,r in enumerate(pd.read_csv(out/f'reproduced_b{b}.csv').itertuples()):
            x=left+(r.x-5.95)*width/1.1;y=bottom+(r.pos+.5)*height/11
            (p.moveTo if i==0 else p.lineTo)(x,y)
        c.drawPath(p,stroke=1,fill=0);c.restoreState()
    c.save();page.merge_page(PdfReader(stream).pages[0]);w=PdfWriter();w.add_page(page);w.write(out/'figure2_reproduction.pdf')
    subprocess.run(['pdftoppm','-singlefile','-r','300','-png',str(out/'figure2_reproduction.pdf'),str(out/'figure2_reproduction')],check=True,capture_output=True)

def plot(results,out):
    m=season();temp=out/'figure1_inputs';shutil.copytree(results/'figure1',temp)
    m.plot(temp,list(range(1,11)),123,None,300,figures_only=True)
    for ext in ('png','pdf'):shutil.move(temp/f'figure1_reproduction.{ext}',out/f'figure1_reproduction.{ext}')
    shutil.rmtree(temp)
    figure2(results/'figure2',out)

def cpconfig(budget=50):
    from boss.config import BOConfig
    cfg=json.loads((ROOT/'results/figure2/run_metadata.json').read_text())['bo_config']
    cfg['bounds']=tuple(tuple(x) for x in cfg['bounds']);cfg['budget']=budget
    cfg['acquisition_maxiter']=300 # Original run predates explicit field; original optimizer cap.
    return BOConfig(**cfg)

def change_compute(out,budget=50):
    from boss.bridge.r_session import RChangePointEvaluator
    from boss.bo.loop import run_boss
    e=RChangePointEvaluator(ROOT/'data/change_point.rda',ROOT/'R/change_point_evaluator.R')
    cfg=cpconfig(budget);result=run_boss(e,cfg);result.trace_frame().to_csv(out/'bo_trace.csv',index=False)
    from dataclasses import asdict
    (out/'run_metadata.json').write_text(json.dumps({'bo_config':asdict(cfg)},indent=2))

def smoke(out):
    m=season();m.GRID=np.linspace(.5,4.5,7);m.BUDGETS=(6,)
    original=m.config
    from dataclasses import replace
    m.config=lambda seed:replace(original(seed),budget=6)
    d=out/'figure1';d.mkdir();m.compute_seed(d,123)
    before=json.loads((d/'seed_123/checkpoint.json').read_text());m.compute_seed(d,123)
    after=json.loads((d/'seed_123/checkpoint.json').read_text())
    assert before['bo']==after['bo'] and before['oracle']==after['oracle']
    m.config=lambda seed:replace(original(seed),budget=7)
    try:m.compute_seed(d,123)
    except ValueError:pass
    else:raise AssertionError('Incompatible checkpoint accepted')
    d=out/'figure2';d.mkdir();change_compute(d,6)
    assert len(pd.read_csv(d/'bo_trace.csv'))==6
    (out/'smoke.json').write_text(json.dumps({'passed':True,'seasonality_resume':True,'checkpoint_mismatch_rejected':True,'full_scientific_rerun':False},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plot','compute','smoke']);p.add_argument('--figure',type=int,choices=[1,2]);p.add_argument('--results-dir',type=Path,default=ROOT/'results');p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args()
    torch.set_num_threads(1)
    if a.command=='compute' and (a.figure is None or a.output is None):p.error('compute requires --figure and --output')
    if a.resume and (a.command!='compute' or a.figure!=1):p.error('--resume applies to figure 1 computation')
    out=a.output or ROOT/'output'
    if a.resume:
        out=writable(out)
        if not (out/'figure1').is_dir():p.error('No figure1 run to resume')
    else:out=fresh(out)
    if a.command=='plot':plot(a.results_dir.resolve(),out)
    elif a.command=='smoke':smoke(out)
    elif a.figure==2:
        d=out/'figure2';d.mkdir();change_compute(d)
    else:
        m=season();d=out/'figure1';d.mkdir(exist_ok=True)
        for seed in [123,*range(1,11)]:m.compute_seed(d,seed)
        m.aggregate(d,list(range(1,11)),123)
    print('Completed:',out)
if __name__=='__main__':main()
