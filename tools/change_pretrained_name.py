from collections import OrderedDict
import torch


# print(model_dict)
pretrained = torch.load('/data2/hyb/SegNetwork_other/SegNeXt-main/pretrained/segnext_small_1024x1024_city_160k.pth')
new_state = dict()

for k, v in pretrained.items():

    if k == 'state_dict':
        for key, value in pretrained[k].items():
            if key[:9] == 'backbone.':
                new_state[key[9:]] = value
            elif key[:11] == 'decode_head.':
                new_state[key[11:]] = value
            else:
                new_state[key] = value
        # new_state = OrderedDict([(key[9:], value) if key[:9] == 'backbone.' or key[:9] == 'decode_head.' else (key, value) for key, value in pretrained[k].items()])

torch.save(new_state, '/data2/hyb/SegNetwork_other/SegNeXt-main/pretrained/re_segnext_small_1024x1024_city_160k.pth')

        # print('change {}'.format(key[9:]))
        # for key, value in pretrained[k].items():
        #     if key[:9] == 'backbone.':
        #         new_state[k][key[9:]] = pretrained[k][key]
        #         new_state[k] = OrderedDict([(key[9:], v) if key[:9] == 'backbone.' else (key, value) for key, value in pretrained[k].items()])
        #         print('change {}'.format(key[9:]))
                # continue

        # keys.append(k)
# i = 0
# for k, v in model_dict.items():
#     if v.size() == pretrained_dict[keys[i]].size():
#          model_dict[k] = pretrained_dict[keys[i]]
#          #print(model_dict[k])
#          i = i + 1
# model.load_state_dict(model_dict)