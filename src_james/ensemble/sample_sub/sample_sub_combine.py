#####
from src_james.ensemble.sample_sub.path import output_dir
from src_james.ensemble.sample_sub.sample_sub1 import sample_sub1
from src_james.ensemble.sample_sub.sample_sub2 import sample_sub2
from src_james.ensemble.sample_sub.sample_sub4 import sample_sub4

sample_sub1 = sample_sub1.reset_index(drop=True).sort_values(by="output_id")
sample_sub2 = sample_sub2.set_index('output_id', drop=True).sort_values(by="output_id")
# sample_sub3 = sample_sub3.set_index('output_id', drop=True).sort_values(by="output_id")
sample_sub4 = sample_sub4.reset_index(drop=True).sort_values(by="output_id")

out1 = sample_sub1["output"].astype(str).values
out2 = sample_sub2["output"].astype(str).values
# out3 = sample_sub3["output"].astype(str).values
out4 = sample_sub4["output"].astype(str).values
merge_output = []

# for o1, o4 in zip(out1, out4):
#     o = o1.strip().split(" ")[:1] + o4.strip().split(" ")[:1]
for o1, o2, o4 in zip(out1, out2, out4):
    o = o1.strip().split(" ")[:1] + o2.strip().split(" ")[:1] + o4.strip().split(" ")[:1]
    o = " ".join(o[:3])
    merge_output.append(o)

sample_sub1["output"] = merge_output
sample_sub1["output"] = sample_sub1["output"].astype(str)
sample_sub1.to_csv(output_dir/"submission.csv", index=False)