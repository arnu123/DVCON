"""
Inspect the LUT's underlying support counts.

For each task, dump the top preferred classes alongside their denominator
(how many images that class appeared in for that task). High score + low
denominator = suspicious. High score + high denominator = trustworthy prior.

Usage:
    python inspect_lut_support.py --coco-tasks-dir ./coco-tasks/annotations

First run gave this output:    
=== Task 1: step_on_something ===
class                score   appears   preferred  trust
chair                 0.79      2120        1677  HIGH
bench                 0.22       230          50  HIGH
couch                 0.22       601         130  HIGH
dining table          0.12      1209         149  HIGH
bed                   0.07       301          22  HIGH
toilet                0.03       200           5  HIGH
suitcase              0.02        60           1  HIGH
laptop                0.01       259           2  HIGH

=== Task 2: sit_comfortably ===
class                score   appears   preferred  trust
couch                 0.91       607         554  HIGH
chair                 0.89      2099        1861  HIGH
toilet                0.68       220         149  HIGH
bench                 0.68       201         136  HIGH
bed                   0.61       302         183  HIGH
dining table          0.03      1230          42  HIGH
bicycle               0.02        65           1  HIGH
motorcycle            0.01        68           1  HIGH

=== Task 3: place_flowers ===
class                score   appears   preferred  trust
vase                  0.86       225         193  HIGH
cup                   0.72       890         640  HIGH
bottle                0.64       880         561  HIGH
wine glass            0.41       199          82  HIGH
bowl                  0.14       745         105  HIGH
hair drier            0.08        12           1  MED
potted plant          0.05        79           4  HIGH
teddy bear            0.00       273           1  HIGH

=== Task 4: get_potatoes_out_of_fire ===
class                score   appears   preferred  trust
baseball bat          1.00       386         385  HIGH
tennis racket         0.99       417         413  HIGH
skis                  0.98       399         390  HIGH
skateboard            0.97       308         300  HIGH
surfboard             0.95       304         289  HIGH
snowboard             0.93       183         170  HIGH
frisbee               0.91       121         110  HIGH
fork                  0.86        98          84  HIGH

=== Task 5: water_plant ===
class                score   appears   preferred  trust
cup                   0.73       913         665  HIGH
wine glass            0.58       256         149  HIGH
bottle                0.58       912         529  HIGH
vase                  0.55       272         149  HIGH
bowl                  0.44       746         330  HIGH
baseball bat          0.04        25           1  HIGH
potted plant          0.02       206           4  HIGH
cake                  0.01       189           1  HIGH

=== Task 6: get_lemon_out_of_tea ===
class                score   appears   preferred  trust
fork                  0.92       515         473  HIGH
spoon                 0.89       535         474  HIGH
scissors              0.78        59          46  HIGH
toothbrush            0.76        89          68  HIGH
knife                 0.50       608         304  HIGH
bottle                0.29      1382         404  HIGH
banana                0.29       138          40  HIGH
wine glass            0.11       379          43  HIGH

=== Task 7: dig_hole ===
class                score   appears   preferred  trust
spoon                 1.00         1           1  LOW
tennis racket         1.00       518         516  HIGH
baseball bat          1.00       467         465  HIGH
skateboard            0.98       373         364  HIGH
surfboard             0.96       336         324  HIGH
skis                  0.95       538         512  HIGH
frisbee               0.95       142         135  HIGH
snowboard             0.94       256         241  HIGH

=== Task 8: open_bottle_of_beer ===
class                score   appears   preferred  trust
scissors              0.73        41          30  HIGH
fork                  0.71        62          44  HIGH
knife                 0.69        75          52  HIGH
spoon                 0.60        25          15  HIGH
bench                 0.43       192          83  HIGH
fire hydrant          0.38        24           9  HIGH
chair                 0.22      1083         239  HIGH
dining table          0.19       734         142  HIGH

=== Task 9: open_parcel ===
class                score   appears   preferred  trust
knife                 0.89       456         406  HIGH
scissors              0.85        91          77  HIGH
toothbrush            0.63       100          63  HIGH
fork                  0.46       381         176  HIGH
spoon                 0.42       389         165  HIGH
skis                  0.25         4           1  LOW
bottle                0.00       916           1  HIGH
hot dog               0.00        49           0  HIGH

=== Task 10: serve_wine ===
class                score   appears   preferred  trust
wine glass            0.91       403         367  HIGH
cup                   0.69      1366         943  HIGH
bowl                  0.02      1077          24  HIGH
vase                  0.02       228           5  HIGH
bottle                0.01      1436           8  HIGH
chair                 0.00       809           1  HIGH
hot dog               0.00        90           0  HIGH
dog                   0.00        82           0  HIGH

=== Task 11: pour_sugar ===
class                score   appears   preferred  trust
spoon                 0.85       499         426  HIGH
knife                 0.71       626         445  HIGH
fork                  0.54       507         272  HIGH
wine glass            0.46       384         176  HIGH
cup                   0.37      1333         499  HIGH
bottle                0.33      1423         472  HIGH
bowl                  0.11      1052         111  HIGH
fire hydrant          0.06        18           1  MED

=== Task 12: smear_butter ===
class                score   appears   preferred  trust
knife                 0.83       657         546  HIGH
spoon                 0.54       536         291  HIGH
toothbrush            0.48        65          31  HIGH
fork                  0.43       514         221  HIGH
scissors              0.24        50          12  HIGH
carrot                0.01       198           2  HIGH
wine glass            0.01       370           2  HIGH
cup                   0.00      1364           1  HIGH

=== Task 13: extinguish_fire ===
class                score   appears   preferred  trust
vase                  0.77       530         406  HIGH
cup                   0.68       622         421  HIGH
bottle                0.58       632         369  HIGH
wine glass            0.50       134          67  HIGH
bowl                  0.49       441         218  HIGH
potted plant          0.02       369           9  HIGH
umbrella              0.01        67           1  HIGH
book                  0.00      1067           1  HIGH

=== Task 14: pound_carpet ===
class                score   appears   preferred  trust
baseball bat          1.00       587         585  HIGH
tennis racket         0.99       638         633  HIGH
skateboard            0.96       357         344  HIGH
skis                  0.95       519         495  HIGH
snowboard             0.92       262         242  HIGH
surfboard             0.92       372         343  HIGH
frisbee               0.81       132         107  HIGH
keyboard              0.40        10           4  MED

So we removed the two LOW trust entries in build_affordance_lut
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


COCO_CAT_ID_TO_CONTIG = {
    1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10, 13:11,
    14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20,
    23:21, 24:22, 25:23, 27:24, 28:25, 31:26, 32:27, 33:28, 34:29,
    35:30, 36:31, 37:32, 38:33, 39:34, 40:35, 41:36, 42:37, 43:38,
    44:39, 46:40, 47:41, 48:42, 49:43, 50:44, 51:45, 52:46, 53:47,
    54:48, 55:49, 56:50, 57:51, 58:52, 59:53, 60:54, 61:55, 62:56,
    63:57, 64:58, 65:59, 67:60, 70:61, 72:62, 73:63, 74:64, 75:65,
    76:66, 77:67, 78:68, 79:69, 80:70, 81:71, 82:72, 84:73, 85:74,
    86:75, 87:76, 88:77, 89:78, 90:79,
}

COCO80 = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

TASKS = ["step_on_something","sit_comfortably","place_flowers","get_potatoes_out_of_fire","water_plant","get_lemon_out_of_tea","dig_hole","open_bottle_of_beer","open_parcel","serve_wine","pour_sugar","smear_butter","extinguish_fire","pound_carpet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-tasks-dir", required=True)
    args = ap.parse_args()

    ann_dir = Path(args.coco_tasks_dir)

    for t_idx, tname in enumerate(TASKS):
        task_id = t_idx + 1
        # Find the train file (Kaggle naming)
        for cand in [f"task_{task_id}.json", f"task_{task_id}_train.json"]:
            p = ann_dir / cand
            if p.exists():
                ann_path = p
                break
        else:
            print(f"!! missing task {task_id}")
            continue

        with open(ann_path) as f:
            data = json.load(f)
        anns = data["annotations"] if isinstance(data, dict) and "annotations" in data else data

        per_image_class = defaultdict(lambda: {"appears": False, "preferred": False})
        for a in anns:
            coco_cat = a.get("COCO_category_id")
            if coco_cat is None or coco_cat not in COCO_CAT_ID_TO_CONTIG:
                continue
            contig = COCO_CAT_ID_TO_CONTIG[coco_cat]
            pref_flag = a.get("category_id", 0)
            key = (a["image_id"], contig)
            per_image_class[key]["appears"] = True
            if pref_flag == 1:
                per_image_class[key]["preferred"] = True

        # Aggregate per class
        appears = defaultdict(int)
        prefers = defaultdict(int)
        for (_img, contig), flags in per_image_class.items():
            if flags["appears"]:
                appears[contig] += 1
            if flags["preferred"]:
                prefers[contig] += 1

        # Top 5 by score, but show support
        results = []
        for contig in appears:
            if appears[contig] == 0:
                continue
            score = prefers[contig] / appears[contig]
            results.append((COCO80[contig], score, appears[contig], prefers[contig]))

        results.sort(key=lambda x: x[1], reverse=True)

        print(f"\n=== Task {task_id}: {tname} ===")
        print(f"{'class':<18}{'score':>8}{'appears':>10}{'preferred':>12}  trust")
        for name, score, n_app, n_pref in results[:8]:
            trust = "HIGH" if n_app >= 20 else ("MED" if n_app >= 5 else "LOW")
            print(f"{name:<18}{score:>8.2f}{n_app:>10}{n_pref:>12}  {trust}")


if __name__ == "__main__":
    main()
