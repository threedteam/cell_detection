import json

with open("/home/user1/文档/idmatchre.txt") as file:
    lines = file.readlines()
    result_1 = {}
    result_2 = {}
    result_3 = {}
    for l in lines:
        l = l.strip("\n")
        x = l.split("cls")[0]
        image_id = x.split("/")[-1]
        n =  l.split("cls")[-1]
        category_id = n.split("|")[0]
        #print(x)
        score = n.split("|")[-1]
        #print(y)
        string_1 = "image_id"
        string_2 = "category_id"
        string_3 = "score"
        result_1[string_1] = image_id
        result_2[string_2] = int(category_id)
        print(int(category_id))
        # result_3[string_3] = float(score)
        # #print(result_3)
        # dict_all = {}
        # dict_all.update(result_1)
        # dict_all.update(result_2)
        # dict_all.update(result_3)
        #
        # #print(dict_all)
        #
        # with open('/home/user1/文档/idmatchre.json', 'a+') as dump_f:
        #     json.dump(dict_all, dump_f)
        #     dump_f.write("\n")




