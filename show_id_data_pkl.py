import pickle

# with open("data/melting_point_id_data.pkl", "rb") as f:
with open("data/viscosity_id_data.pkl", "rb") as f:
    data = pickle.load(f)

print("记录条数:", len(data))
print("示例一条记录的字段:")
print(data[0].keys())
print()
print("第一条记录内容:")
for k,v in data[0].items():
    print(k, "=", v)
