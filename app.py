import streamlit as st
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import matplotlib_fontja

# ==========================================
# 1. 関数定義
# ==========================================

def process_cfd_files(master_file, cfd_files, rho, cp, threshold):
    # --- マスタ読み込み ---
    try:
        master_file.seek(0)
        rules_df = pd.read_csv(master_file, encoding='cp932', header=0)
    except Exception as e:
        return None, None, None, [f"❌ マスタファイル読み込みエラー: {e}"]

    opening_results_list = []
    logs = []

    # --- 各CFDファイルをループ処理 ---
    total_files = len(cfd_files)
    progress_bar = st.progress(0)

    for i, uploaded_file in enumerate(cfd_files):
        progress_bar.progress((i + 1) / total_files)
        
        file_name = uploaded_file.name
        file_key = os.path.splitext(file_name)[0]
        parts = file_key.split('_')

        found_rule = None
        detected_axis = None

        # --- (A) 軸情報の抽出 ---
        try:
            uploaded_file.seek(0)
            # 先頭の数行だけ読んで軸を探す
            df_temp = pd.read_csv(uploaded_file, skiprows=2, nrows=1, encoding='cp932')
            
            # 「流量算出面」という文字列を含む列を探す
            axis_col = [c for c in df_temp.columns if '流量算出面' in str(c)]

            if axis_col:
                raw_axis_value = str(df_temp[axis_col[0]].iloc[0]).strip()
                if raw_axis_value:
                    detected_axis = raw_axis_value[0].lower() # X, Y, Z -> x, y, z
            else:
                logs.append(f"⚠️ {file_name}: '流量算出面' 列が見つかりません。")
        except Exception as e:
            logs.append(f"⚠️ {file_name}: 軸抽出失敗 ({e})")

        # --- (B) マスタファイルとの照合 ---
        if detected_axis:
            possible_pairs = list(itertools.combinations(parts, 2))
            for pair in possible_pairs:
                room1, room2 = pair[0], pair[1]

                # マスタ照合ロジック
                axis_match = (rules_df.iloc[:, 0].astype(str).str.lower() == detected_axis)
                room_match_1 = (rules_df['Plus_Room'] == room1) & (rules_df['Minus_Room'] == room2)
                room_match_2 = (rules_df['Plus_Room'] == room2) & (rules_df['Minus_Room'] == room1)

                rule_match = rules_df[axis_match & (room_match_1 | room_match_2)]

                if not rule_match.empty:
                    found_rule = rule_match.iloc[0]
                    # タイブレークが必要な場合はここで処理（今回は最初の1つを採用）
                    break
        
        if found_rule is None:
            logs.append(f"⚠️ スキップ: '{file_name}' (軸:{detected_axis}) - マスタ不一致")
            continue

        # --- (C) 全データ読み込みと計算 ---
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, skiprows=2, encoding='cp932') # 基本はcp932(Shift-JIS)
            
            flow_col, temp_col = '流量[m3/h]', 'スカラー量[℃]'
            
            # 数値変換
            df[flow_col] = pd.to_numeric(df[flow_col], errors='coerce')
            df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
            df.dropna(subset=[flow_col, temp_col], inplace=True)

            # 熱計算
            df['heat_kjh'] = df[flow_col] * rho * cp * df[temp_col]
            net_heat_watt = df['heat_kjh'].sum() * 1000 / 3600
            
            # 流量計算
            gross_positive_flow = df[df[flow_col] > 0][flow_col].sum()
            gross_negative_flow = df[df[flow_col] < 0][flow_col].sum()

            opening_results_list.append({
                '開口部': file_key,
                '方向': detected_axis,
                'Plus_Room': found_rule['Plus_Room'],
                'Minus_Room': found_rule['Minus_Room'],
                '総プラス流量[m3/h]': gross_positive_flow,
                '総マイナス流量[m3/h]': gross_negative_flow,
                '移動熱量[W]': net_heat_watt
            })

        except Exception as e:
            logs.append(f"❌ 計算エラー: {file_name} ({e})")

    # 結果をDataFrame化
    if not opening_results_list:
        return None, None, None, logs
    
    results_df = pd.DataFrame(opening_results_list)

    # --- 集計処理 (関数内で実行) ---
    
    # 1. 熱収支集計
    heat_movements = []
    # 移動熱量がプラス
    df_heat_pos = results_df[results_df['移動熱量[W]'] > 0]
    heat_movements.append(pd.DataFrame({'室名': df_heat_pos['Minus_Room'], '方向': '流出', '熱量[W]': df_heat_pos['移動熱量[W]']}))
    heat_movements.append(pd.DataFrame({'室名': df_heat_pos['Plus_Room'], '方向': '流入', '熱量[W]': df_heat_pos['移動熱量[W]']}))
    # 移動熱量がマイナス
    df_heat_neg = results_df[results_df['移動熱量[W]'] < 0]
    heat_movements.append(pd.DataFrame({'室名': df_heat_neg['Plus_Room'], '方向': '流出', '熱量[W]': df_heat_neg['移動熱量[W]'].abs()}))
    heat_movements.append(pd.DataFrame({'室名': df_heat_neg['Minus_Room'], '方向': '流入', '熱量[W]': df_heat_neg['移動熱量[W]'].abs()}))
    
    heat_df = pd.concat(heat_movements).groupby(['室名', '方向'])['熱量[W]'].sum().unstack(fill_value=0)
    room_heat_summary_df = pd.DataFrame({
        '総流出熱量[W]': heat_df.get('流出', 0),
        '総流入熱量[W]': heat_df.get('流入', 0),
        '処理熱量[W]': heat_df.get('流出', 0) - heat_df.get('流入', 0)
    }).reset_index()

    # 2. 風量収支集計
    flow_movements = []
    flow_movements.append(pd.DataFrame({'室名': results_df['Minus_Room'], '方向': '流出', '流量[m3/h]': results_df['総プラス流量[m3/h]']}))
    flow_movements.append(pd.DataFrame({'室名': results_df['Plus_Room'], '方向': '流入', '流量[m3/h]': results_df['総プラス流量[m3/h]']}))
    flow_movements.append(pd.DataFrame({'室名': results_df['Plus_Room'], '方向': '流出', '流量[m3/h]': results_df['総マイナス流量[m3/h]'].abs()}))
    flow_movements.append(pd.DataFrame({'室名': results_df['Minus_Room'], '方向': '流入', '流量[m3/h]': results_df['総マイナス流量[m3/h]'].abs()}))

    flow_df = pd.concat(flow_movements).groupby(['室名', '方向'])['流量[m3/h]'].sum().unstack(fill_value=0)
    room_flow_summary_df = pd.DataFrame({
        '総流出流量[m3/h]': flow_df.get('流出', 0),
        '総流入流量[m3/h]': flow_df.get('流入', 0),
        '風量収支[m3/h]': flow_df.get('流出', 0) - flow_df.get('流入', 0)
    }).reset_index()

    return results_df, room_heat_summary_df, room_flow_summary_df, logs

def create_heat_chart(room_heat_summary_df, fig_width, fig_height, font_size, y_max, custom_colors, show_legend, category_map):
    # --- データ準備 ---
   if "暖房" in mode:
        label_passive = "各室熱損失"
        label_active = "投入熱量"
        passive = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] < 0].set_index('室名')['処理熱量[W]'].abs()
        active = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] > 0].set_index('室名')['処理熱量[W]']
   else: 
        label_passive = "各室負荷"
        label_active = "処理熱量"
        passive = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] > 0].set_index('室名')['処理熱量[W]']
        active = room_heat_summary_df[room_heat_summary_df['処理熱量[W]'] < 0].set_index('室名')['処理熱量[W]'].abs
        
    plot_df_base = pd.DataFrame({label_passive: passive , label_active: active}).T.fillna(0)

    # --- 並べ替えロジック (引数の category_map を使用) ---
    # マップ内のリストを展開して、並べ替え順序リストを作成
    desired_order = []
    for rooms in category_map.values():
        desired_order.extend(rooms)
    
    current_columns = plot_df_base.columns.tolist()
    
    # マップにある部屋を優先し、マップにない部屋は後ろに追加
    ordered_columns = [col for col in desired_order if col in current_columns]
    remaining_columns = [col for col in current_columns if col not in desired_order]
    final_column_order = ordered_columns + remaining_columns
    
    # データフレームを並べ替え
    plot_df = plot_df_base[final_column_order]
    
    # --- 色の適用 ---
    colors = []
    default_color = '#AAAAAA'
    for room in final_column_order:
        colors.append(custom_colors.get(room, default_color))

    # --- 描画 ---
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8, legend=False)

    # --- 見た目調整 ---
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7, color='#cccccc')
    ax.grid(axis='x', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.tick_params(axis='y', length=0, labelsize=font_size)
    ax.tick_params(axis='x', length=0)
    plt.xticks(rotation=0, fontsize=font_size)
    plt.ylabel('処理熱量[W]', fontsize=font_size)
    
    if y_max > 0:
        ax.set_ylim(0, y_max)

    plt.axhline(0, color='black', linewidth=0.8)

    # --- バーの数値ラベル ---
    for i, container in enumerate(ax.containers):
        labels = [f"{v:,.0f}" if v > 0 else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', color='black', fontsize=font_size*0.8, fontweight='bold')

    # --- 凡例の作成 (動的生成) ---
    if show_legend:
        handles, labels_legend = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        dummy_handle = mpatches.Patch(visible=False)

        # マップの定義順(逆順)に凡例グループを作成
        for category_name, rooms_in_category in reversed(category_map.items()):
            category_handles_labels = []
            
            # 各カテゴリ内の部屋順(逆順)に処理
            for room_name in reversed(rooms_in_category):
                if room_name in labels_legend:
                    index = labels_legend.index(room_name)
                    category_handles_labels.append((handles[index], f"  {room_name}"))
            
            if category_handles_labels:
                # カテゴリ名を表示（空文字以外）
                if category_name:
                    new_handles.append(dummy_handle)
                    new_labels.append(f"--- {category_name} ---")
                
                for handle, label in category_handles_labels:
                    new_handles.append(handle)
                    new_labels.append(label)

        # マップに含まれなかった残りの部屋（未分類）
        remaining_items = [(handles[i], f"  {labels_legend[i]}") for i, label in enumerate(labels_legend) if label not in desired_order]
        if remaining_items:
            new_handles.append(dummy_handle)
            new_labels.append("▼ 未分類")
            for handle, label in remaining_items:
                new_handles.append(handle)
                new_labels.append(label)

        ax.legend(handles=new_handles, labels=new_labels, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=font_size*0.9)

    total_pos = pos_data.sum()
    total_neg = neg_data.sum()
    
    return fig, total_pos, total_neg

# ==========================================
# 2. アプリケーション UI
# ==========================================

st.set_page_config(page_title="CFD 熱量分析ツール", layout="wide")

st.title("CFD 熱量分配 & 風量バランス分析")
st.markdown("FlowDesignerでCSV出力した開口部やエアコンの吹出・吸込口の温度・速度から処理熱量を部屋ごとに集計するツールです")

# --- サイドバー設定 ---
with st.sidebar:
    st.header("1. 解析設定")
    mode = st.radio("モード, ["冷房","暖房"])
    st.divider()
    
    st.header("2. 定数設定")
    rho = st.number_input("空気密度 ρ [kg/m3]", value=1.20)
    cp = st.number_input("比熱 Cp [J/g・K]", value=1.006, format="%.3f")
    threshold = st.number_input("風量収支許容誤差 [m3/h]", value=1.0)
    
    st.header("3. 分析ファイル")
    st.info("マスタファイル (各室の位置関係を記述したファイル)をドラッグ＆ドロップまたはブラウズ")
    master_file = st.file_uploader("マスタファイル", type="csv")
    st.markdown("---")
    st.info("FDで書きだした開口部のCSVを全てドラッグ＆ドロップまたはブラウズ")
    cfd_files = st.file_uploader("CFD解析結果 (複数選択)", type="csv", accept_multiple_files=True)

# --- メイン処理 ---

# 1. セッションステート（記憶領域）の初期化
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
    st.session_state['results_df'] = None
    st.session_state['room_heat_df'] = None
    st.session_state['room_flow_df'] = None
    st.session_state['logs'] = []

# 2. 解析ボタン（押されたら計算して保存するだけ）
if st.button("解析実行", type="primary"):
    if not master_file or not cfd_files:
        st.warning("マスタファイルとCFD解析結果の両方をアップロードしてください。")
    else:
        with st.spinner("計算中"):
            # 計算実行
            results_df, room_heat_df, room_flow_df, logs = process_cfd_files(master_file, cfd_files, rho, cp, threshold)
            
            if results_df is not None:
                # 結果をセッションステートに保存
                st.session_state['results_df'] = results_df
                st.session_state['room_heat_df'] = room_heat_df
                st.session_state['room_flow_df'] = room_flow_df
                st.session_state['logs'] = logs
                st.session_state['analyzed'] = True
                st.success("解析完了")
            else:
                st.session_state['logs'] = logs
                st.error("有効なデータが作成されませんでした。ログを確認してください。")

# 3. データがある場合の表示処理（スライダーを動かしてもここは再実行される）
if st.session_state['analyzed']:
    # 保存されたデータを読み出し
    results_df = st.session_state['results_df']
    room_heat_df = st.session_state['room_heat_df']
    room_flow_df = st.session_state['room_flow_df']
    logs = st.session_state['logs']

    # ログの表示
    with st.expander("エラー・警告", expanded=False):
        for log in logs:
            if "❌" in log: st.error(log)
            elif "⚠️" in log: st.warning(log)
            else: st.info(log)

    # --- タブによる表示切り替え ---
    tab1, tab2, tab3 = st.tabs(["風量収支チェック", "熱量分配グラフ", "計算詳細"])

    # --- Tab 1: 風量バランス ---
    with tab1:
        st.subheader("風量収支チェック")
        st.caption(f"許容誤差: ±{threshold} m3/h")
        
        warning_count = 0
        for index, row in room_flow_df.iterrows():
            room = row['室名']
            balance = row['風量収支[m3/h]']
            
            if balance > threshold:
                st.error(f"⚠️ {room}: 流出過多 (流入不足) +{balance:.2f} m3/h")
                warning_count += 1
            elif balance < -threshold:
                st.error(f"⚠️ {room}: 流入過多 (流出不足) {balance:.2f} m3/h")
                warning_count += 1
            else:
                st.success(f"{room}: OK ({balance:+.2f} m3/h)")
        
        if warning_count == 0:
            # st.balloons()  # 風船を飛ばしたい場合オンに
            st.info("✅ 全室で風量収支が許容値以下")

 # --- Tab 2: グラフ ---
    with tab2:
        st.subheader("各室およびエアコンの空調処理熱量")

        # データに含まれる全室名を取得
        all_rooms = sorted(room_heat_df['室名'].unique())

        # --- グラフ設定エリア ---
        with st.expander("グラフをカスタマイズする", expanded=False):
            
            # --- 上段: グルーピング設定 ---
            st.markdown("#### 凡例グループと並び順")
            st.caption("カテゴリ名を入力し、所属する部屋を選択してください。選択順にグラフの下側から積み上がります。")
            
            # デフォルト設定 (初回のみ使用)
            default_categories_list = [
                ("１階", ["1階", "床下", "LDK", "洗面室", "和室"]),
                ("２階", ["R3", "R2", "R1", "廊下", "SR", "小屋裏"]),
                ("空調機", ["AC"])
            ]

            #カテゴリー数定義
            num_categories = st.number_input("カテゴリー数", min_value = 1, max_value = 10, value = 3, step = 1)
            # ユーザー設定用のコンテナ
            custom_category_map = {}
            
            #レイアウト用カラム作成（3列グリッド)
            cols_cat = st.columns(3)

            #カテゴリー入力欄作成
            for i in range(num_categories):
                with cols_cat[i % 3]:
                    if i < len(default_categories_list):
                        def_name = default_categories_list[i][0]
                        def_rooms = default_categories_list[i][1]
                        def_rooms = [r for r in def_rooms if r in all_rooms]
                    else:
                        def_name  = (f"グループ{i+1}")
                        def_rooms = []

                    #カテゴリ名入力
                    cat_name = st.text_input(f"カテゴリ名{i+1}",value=def_name, key=f"cat_name_{i}")
                    # 部屋選択
                    selected_rooms = st.multiselect(
                        f"{cat_name} の部屋", 
                        options=all_rooms, 
                        default=def_rooms,
                        key=f"cat_rooms_{i}"
                    )
                    
                    # カテゴリ名が空でなければマップに追加
                    if cat_name and selected_rooms:
                        custom_category_map[cat_name] = selected_rooms

            st.divider()

            # --- 下段: 見た目設定 ---
            st.markdown("#### グラフ体裁")
            col_ui1, col_ui2, col_ui3 = st.columns(3)
            
            with col_ui1:
                st.markdown("**サイズ設定**")
                fig_w = st.number_input("横幅 (inch)", value=6.0, step=0.5)
                fig_h = st.number_input("高さ (inch)", value=10.0, step=0.5)
            
            with col_ui2:
                st.markdown("**表示設定**")
                font_size = st.slider("文字サイズ", 8, 40, 14)
                y_max = st.number_input("Y軸の最大値 (0で自動)", value=0, step=100)
                show_legend = st.checkbox("凡例を表示する", value=True)

            with col_ui3:
                st.markdown("**色の設定**")
                default_colors = {
                    "LDK": "#FF7F50", "1階": "#FF7F50", "2階": "#0000FF", "廊下": "#9370DB",
                    "R1": "#6495ED", "R2": "#FFA500", "R3": "#32CD32", "床下": "#D3D3D3",
                    "AC": "#87CEEB", "小屋裏": "#ADFF2F", "洗面室": "#40E0D0",
                    "和室": "#BDB76B", "SR": "#FFFF00",
                }
                
                custom_colors = {}
                if st.checkbox("色を個別に変更する"):
                    for room in all_rooms:
                        initial = default_colors.get(room, "#AAAAAA")
                        picked = st.color_picker(f"{room}", value=initial, key=f"color_{room}")
                        custom_colors[room] = picked
                else:
                    custom_colors = default_colors

        # --- グラフ描画実行 ---
        try:
            # 引数に custom_category_map を追加
            fig, total_passive, total_active = create_heat_chart(
                room_heat_df, fig_w, fig_h, font_size, y_max, custom_colors, show_legend, custom_category_map, mode
            )
            
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            if "暖房"　in mode:
                label_left = "各室熱損失合計"
                label_right = "投入熱量"
            else:
                label_left = "各室熱負荷合計"
                label_right = "処理熱量"
            col1.metric(label_left, f"{total_passive:,.1f} W")
            col2.metric(label_right, f"{total_active:,.1f} W")
            
            # 画像ダウンロード
            img = io.BytesIO()
            fig.savefig(img, format='svg', bbox_inches='tight')
            st.download_button("グラフをSVGで保存", img, "heat_balance.svg", "image/svg+xml")
            
        except Exception as e:
            st.error(f"グラフ作成エラー: {e}")

    # --- Tab 3: 計算詳細 ---
    with tab3:
        # ダウンロードボタンもここに追加しておきます
        st.markdown("### 📥 データダウンロード")
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        col_dl1.download_button("表1 (開口部風量・移動熱量)", results_df.to_csv(index=False).encode('shift_jis'), "results_raw.csv")
        col_dl2.download_button("表2 (処理熱量)", room_heat_df.to_csv(index=False).encode('shift_jis'), "results_heat.csv")
        col_dl3.download_button("表3 (風量収支)", room_flow_df.to_csv(index=False).encode('shift_jis'), "results_flow.csv")
        st.divider()

        st.markdown("### (表1) 開口部別 風量・移動熱量")
        st.dataframe(results_df)
        
        st.markdown("### (表2) 室別 処理熱量")
        st.dataframe(room_heat_df)
        
        st.markdown("### (表3) 室別 風量収支")
        st.dataframe(room_flow_df)

else:

        st.error("有効なデータが作成されませんでした。ログを確認してください。")




