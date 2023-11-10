import json
import xarray as xr
import collections
import numpy as np


def process_nafc_assignment_json(asn_json: dict):
    """
    Asn_json
    :param asn_json:
    :return:
    """
    answer_text = asn_json['Answer']
    answer_text = answer_text.split('<FreeText>')[1].split('</FreeText>')[0]

    data = (json.loads(answer_text))
    ds = process_session_data(
        session_data=data['trials'],
    )

    return ds


def compute_bonus_amount(ds:xr.Dataset):
    ngt = ds.perf.sum()
    bonus_usd_earned = (ds.bonus_usd_if_correct * ds.perf).sum()
    print(f'{bonus_usd_earned} USD earned ({int(ngt)} correct)')
    return bonus_usd_earned

def process_session_data(session_data:list):

    data_vars = collections.defaultdict(list)
    coords = collections.defaultdict(list)


    data_var_names = [
        'perf',
        'i_choice',
        'reaction_time_msec',
    ]
    coord_var_names = [
        'i_correct_choice',
        'bonus_usd_if_correct',
        'timestamp_start',
        'rel_timestamp_response',
        'choice_duration_msec',
        'stimulus_image_url',
        'choice_image_urls',
        'query_string',
        'stimulus_duration_msec',
        'post_stimulus_delay_duration_msec',
        'pre_choice_lockout_delay_duration_msec',
        'keep_stimulus_on',
        'stimulus_width_px',
        'choice_width_px',
        'monitor_width_px',
        'monitor_height_px',
        'bonus_usd_if_correct',
        'mask_image_url',
        'mask_duration_msec',
    ]
    data_var_names = sorted(set(data_var_names))
    coord_var_names = sorted(set(coord_var_names))

    for jspsych_trial in session_data:
        if 'trial_outcome' not in jspsych_trial:
            continue

        dat = jspsych_trial['trial_outcome']

        for name in data_var_names:
            if name in dat:
                data_vars[name].append(dat[name])
            else:
                data_vars[name].append(np.nan)

        for name in coord_var_names:
            if name in dat:
                coords[name].append(dat[name])
            else:
                coords[name].append(np.nan)

    for k in data_vars:
        data_vars[k] = (['obs'], data_vars[k])
    for k in coords:
        if k == 'choice_image_urls':
            max_nchoices = max([len(x) for x in coords[k]])
            # Pad with '' to make all lists the same length
            for i in range(len(coords[k])):
                coords[k][i] += [''] * (max_nchoices - len(coords[k][i]))
            coords[k] = (['obs', 'choice_slot'], coords[k])
        else:
            coords[k] = (['obs'], coords[k])


    if len(data_vars) == 0:
        return None
    ds = xr.Dataset(
        data_vars = data_vars,
        coords = coords,
    )

    pvalues = [p  if p is not None else np.nan for p in ds['perf'].values]
    ds['perf'].values[:] = pvalues

    ds['obs'] = np.arange(len(ds['obs']))
    return ds

