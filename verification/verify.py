"""Compare freshly rendered figures and spline inputs with report targets."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from PIL import Image
root=Path(__file__).resolve().parents[1];out=Path(sys.argv[1]).resolve();report={}
for i in (1,2):
 a=np.array(Image.open(root/f'verification/expected/figure{i}_reproduction.png').convert('RGB'));b=np.array(Image.open(out/f'figure{i}_reproduction.png').convert('RGB'))
 assert a.shape==b.shape
 delta=np.abs(a.astype(float)-b)
 report[f'figure{i}']={'shape':list(a.shape),'pixel_identical':bool(np.array_equal(a,b)),'mean_absolute_RGB_difference':float(delta.mean()),'different_pixels':int(np.any(delta>0,axis=2).sum())}
 assert delta.mean()<.001
for budget in (20,50):
 a=pd.read_csv(root/f'verification/expected/reproduced_b{budget}.csv');b=pd.read_csv(out/f'reproduced_b{budget}.csv');np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
report.update(splines_match=True,full_scientific_recomputation=False)
print(json.dumps(report,indent=2))
