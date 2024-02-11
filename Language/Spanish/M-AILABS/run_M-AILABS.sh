#!/bin/bash
# FEMALE
echo "Running for FEMALE corpus..."
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/female/karen_savage/angelina/wavs -t /mnt/corpus/M-AILABS/female/karen_savage/angelina/metadata_mls.json > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_female.log 2>&1 &

# MIX
echo "Running for MIX corpus..."
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/mix/la_condenada/wavs -t /mnt/corpus/M-AILABS/mix/la_condenada/metadata_mls.json > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_mix_la_condenada.log 2>&1 &
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/mix/don_quijote/wavs  -t /mnt/corpus/M-AILABS/mix/don_quijote/metadata_mls.json > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_mix_don_quijote.log 2>&1 &

# MALE
echo "Running for MALE corpus..."
# TUX
echo "Running for TUX corpus..."
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/male/tux/bailen/wavs -t /mnt/corpus/M-AILABS/male/tux/bailen/metadata_mls.json  > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_male_tux_bailen.log 2>&1 &
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/male/tux/el_19_de_marzo_y_el_2_de_nayo/wavs -t /mnt/corpus/M-AILABS/male/tux/el_19_de_marzo_y_el_2_de_nayo/metadata_mls.json  > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_male_tux_el_19_de_marzo_y_el_2_de_nayo.log 2>&1 &
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/male/tux/eneida/wavs -t /mnt/corpus/M-AILABS/male/tux/eneida/metadata_mls.json  > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_male_tux_eneida.log 2>&1 &
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/male/tux/la_batalla_de_los_arapiles/wavs -t /mnt/corpus/M-AILABS/male/tux/la_batalla_de_los_arapiles/metadata_mls.json  > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_male_tux_la_batalla_de_los_arapiles.log 2>&1 &
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/male/tux/la_corte_de_carlos_iv/wavs -t /mnt/corpus/M-AILABS/male/tux/la_corte_de_carlos_iv/metadata_mls.json  > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_male_tux_la_corte_de_carlos_iv.log 2>&1 &
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/male/tux/napoleon_en_chamartin/wavs -t /mnt/corpus/M-AILABS/male/tux/napoleon_en_chamartin/metadata_mls.json  > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_male_tux_napoleon_en_chamartin.log 2>&1 &
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/male/tux/trafalgar/wavs -t /mnt/corpus/M-AILABS/male/tux/trafalgar/metadata_mls.json  > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_male_tux_trafalgar.log 2>&1 &

# VICTOR VILLARRAZA
echo "Running for VICTOR VILLARRAZA corpus..."
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/male/victor_villarraza/cuentos_clasicos_del_norte/wavs -t /mnt/corpus/M-AILABS/male/victor_villarraza/cuentos_clasicos_del_norte/metadata_mls.json  > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_male_victor_villarraza_cuentos_clasicos_del_norte.log 2>&1 &
nohup python3 -m M-AILABS.main -a /mnt/corpus/M-AILABS/male/victor_villarraza/la_dama_de_las_camelias/wavs -t /mnt/corpus/M-AILABS/male/victor_villarraza/la_dama_de_las_camelias/metadata_mls.json  > /home/aholab/santi/Documents/audio_process/Language/Spanish/M-AILABS/m-ailabs_male_victor_villarraza_la_dama_de_las_camelias.log 2>&1 &