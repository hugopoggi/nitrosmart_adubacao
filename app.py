import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np
from babel.dates import format_date
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="NitroSmart - Adubação Inteligente", layout="wide")

LOCAIS_PREDETERMINADOS = {
    "Usina Tanabi - SP": {"lat": -20.489485534431246, "lon": -49.54360768974638, "nome_curto": "Usina Tanabi"},
    "Usina Cruz Alta - SP": {"lat": -20.692502938632963, "lon": -49.10884492757553, "nome_curto": "Usina Cruz Alta"},
}

RAIO_PREVISAO_KM = 9


@st.cache_resource
def carregar_modelos():
    try:
        model = joblib.load('modelo_farol_adubacao_xgb.pkl')
        le = joblib.load('label_encoder.pkl')
        model_cols = joblib.load('model_columns.pkl')
        return model, le, model_cols
    except FileNotFoundError:
        st.error("ERRO: Arquivos de modelo (.pkl) não encontrados.")
        return None, None, None


model, le, model_cols = carregar_modelos()


@st.cache_data(ttl=3600)
def buscar_previsao_por_coords(lat, lon):
    try:
        meteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weathercode,temperature_2m_max,precipitation_sum,windspeed_10m_max&hourly=relativehumidity_2m&timezone=America/Sao_Paulo&forecast_days=10"
        meteo_res = requests.get(meteo_url);
        meteo_data = meteo_res.json();
        return meteo_data
    except Exception as e:
        st.error(f"Erro ao buscar a previsão do tempo: {e}");
        return None


def processar_previsao(previsao, params):
    df = pd.DataFrame(previsao['daily']);
    df.rename(columns={'time': 'data', 'temperature_2m_max': 'temperatura_max_c',
                       'precipitation_sum': 'precipitacao_diaria_mm', 'windspeed_10m_max': 'velocidade_vento_kmh'},
              inplace=True)
    for key, value in params.items(): df[key] = value
    df['Chuva_Acumulada_72h_Pos'] = df['precipitacao_diaria_mm'].shift(-1).rolling(window=3,
                                                                                   min_periods=1).sum().fillna(0)
    df['Armadilha_Chuva_Insuficiente'] = (
            (df['precipitacao_diaria_mm'].shift(-1) >= 1) & (df['precipitacao_diaria_mm'].shift(-1) <= 10) & (
            df['temperatura_max_c'].shift(-2) > 25)).fillna(False)

    def map_weathercode(code):
        if code in [0, 1]: return 'Ensolarado';
        if code in [2, 3]: return 'Nuvens Esparsas'
        if code in [45, 48]: return 'Nublado'
        if code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: return 'Chuva Leve'
        return 'Chuva Forte'

    df['tipo_ceu'] = df['weathercode'].apply(map_weathercode)
    df['umidade_relativa_ar_percent'] = 70;
    df['Umidade_Palhada_Inferida'] = np.where(df['precipitacao_diaria_mm'] > 5, 'Molhada',
                                              np.where(df['precipitacao_diaria_mm'] > 0, 'Umida', 'Seca'));
    df['Tempo_ate_Chuva_de_Incorporacao'] = 120
    df['risco_volatilizacao'] = (df['temperatura_max_c'] * df['velocidade_vento_kmh']) / (
            df['umidade_relativa_ar_percent'] + 1)
    df['palha_umida_com_calor'] = ((df['Umidade_Palhada_Inferida'] == 'Umida') & (df['temperatura_max_c'] > 28)).astype(
        int)
    return df


def gerar_resultado_final(farol_modelo, cenario):
    temp = cenario['temperatura_max_c'];
    chuva_72h = cenario['Chuva_Acumulada_72h_Pos'];
    armadilha = cenario['Armadilha_Chuva_Insuficiente'];
    umidade_palha = cenario['Umidade_Palhada_Inferida'];
    insumo = cenario['tipo_insumo'];
    precipitacao_dia = cenario['precipitacao_diaria_mm']
    if armadilha and insumo != 'Vinhaca Enriquecida': return "VERMELHO", "ARMADILHA DE VOLATILIZAÇÃO", "Chuva fraca prevista irá dissolver a ureia, e o sol forte seguinte causará perdas massivas. PROIBIR APLICAÇÃO."
    if precipitacao_dia > 25: return "VERMELHO", "RISCO DE ENXURRADA", "Chuva excessiva causará escorrimento superficial e lixiviação."
    if umidade_palha == 'Umida' and temp > 28 and insumo != 'Vinhaca Enriquecida': return "VERMELHO", "BOMBA DE VOLATILIZAÇÃO", "Palha úmida com calor ativa a urease, causando altas perdas."
    if farol_modelo == 'VERMELHO':
        return "VERMELHO", "CONDIÇÕES CRÍTICAS", "Fatores combinados desaconselham a aplicação."
    elif farol_modelo == 'AMARELO':
        if temp > 30 and chuva_72h < 15: return "AMARELO", "RISCO DE PERDA POR CALOR", "Temperaturas altas e sem chuva de incorporação. Recomenda-se fortemente UREIA+NBPT."
        if insumo == 'Vinhaca Enriquecida' and precipitacao_dia > 10: return "AMARELO", "RISCO DE ESCOAMENTO", "Chuva moderada em áreas inclinadas pode causar perdas de vinhaça."
        return "AMARELO", "CONDIÇÕES DE ATENÇÃO", "O cenário não é ideal. Avalie os riscos e considere tecnologias protetoras."
    elif farol_modelo == 'VERDE':
        if insumo == 'Vinhaca Enriquecida': return "VERDE", "INFILTRAÇÃO IDEAL", "Condições perfeitas para fertirrigação com máxima absorção e mínimo risco."
        if chuva_72h >= 15: return "VERDE", "JANELA DE OURO", "Chuva de incorporação garantida. Condições ideais para máxima eficiência."
        if temp < 26 and cenario[
            'tipo_ceu'] == 'Nublado': return "VERDE", "CONDIÇÕES SEGURAS", "Temperaturas amenas e céu nublado minimizam drasticamente as perdas."
        return "VERDE", "CONDIÇÕES FAVORÁVEIS", "Baixo risco e boa janela para adubação."
    return farol_modelo, "", ""


def gerar_resumo_executivo(df_previsao):
    dias_verdes = df_previsao[df_previsao['Farol'] == 'VERDE'];
    dias_amarelos = df_previsao[df_previsao['Farol'] == 'AMARELO']
    if not dias_verdes.empty:
        data_obj = pd.to_datetime(dias_verdes['data'].iloc[0]);
        primeiro_dia_verde = format_date(data_obj, "EEEE, dd/MM", locale='pt_BR').capitalize()
        return f"✅ **Recomendação Principal:** A melhor janela de oportunidade começa na **{primeiro_dia_verde}**."
    elif not dias_amarelos.empty:
        data_obj = pd.to_datetime(dias_amarelos['data'].iloc[0]);
        primeiro_dia_amarelo = format_date(data_obj, "EEEE, dd/MM", locale='pt_BR').capitalize()
        return f"⚠️ **Recomendação Principal:** Nenhuma janela ideal encontrada. Condições de atenção começam na **{primeiro_dia_amarelo}**."
    else:
        return "❌ **Recomendação Principal:** Nenhuma janela de aplicação recomendada para os próximos 10 dias."


st.title("🚜 NitroSmart - Adubação Inteligente")

with st.sidebar:
    st.header("Parâmetros da Análise")
    local_selecionado = st.selectbox("1. Selecione a Unidade", list(LOCAIS_PREDETERMINADOS.keys()))
    tipo_insumo = st.selectbox("2. Tipo de Insumo", ['Ureia Comum', 'Ureia+NBPT', 'Vinhaca Enriquecida'])
    textura_solo = st.selectbox("3. Textura do Solo", ['Argiloso', 'Siltoso', 'Arenoso'])
    volume_palhada_t_ha = st.slider("4. Volume de Palhada (ton/ha)", 5, 20, 12)
    if volume_palhada_t_ha > 15:
        st.caption("ℹ️ Nível alto: Maior risco de perdas de ureia.")
    elif volume_palhada_t_ha < 8:
        st.caption("ℹ️ Nível baixo: Maior exposição do solo ao sol.")
    else:
        st.caption(" ")
    declividade = st.selectbox("5. Declividade do Terreno", ['Plana', 'Levemente Inclinada', 'Inclinada'])

    analisar = st.button("Analisar", use_container_width=True, type="primary")

if not analisar:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Analisar' para gerar a previsão.")

if analisar:
    if model is None: st.stop()
    params_usuario = {"tipo_insumo": tipo_insumo, "textura_solo": textura_solo,
                      "volume_palhada_t_ha": volume_palhada_t_ha, "declividade": declividade}
    coords = LOCAIS_PREDETERMINADOS[local_selecionado]
    previsao_data = buscar_previsao_por_coords(coords['lat'], coords['lon'])

    if previsao_data:
        with st.spinner(f'Buscando previsão para {local_selecionado} e aplicando IA...'):
            df_previsao = processar_previsao(previsao_data, params_usuario)
            df_encoded = pd.get_dummies(df_previsao);
            df_processed = df_encoded.reindex(columns=model_cols, fill_value=0)
            previsoes_num = model.predict(df_processed);
            df_previsao['Farol_Modelo'] = le.inverse_transform(previsoes_num)
            resultados_finais = [gerar_resultado_final(row['Farol_Modelo'], row) for index, row in
                                 df_previsao.iterrows()]
            df_previsao['Farol'], df_previsao['Insight'], df_previsao['Dica'] = zip(*resultados_finais)

            st.header(f"Resultados da Análise para {local_selecionado}")
            resumo = gerar_resumo_executivo(df_previsao)
            st.markdown(f"### {resumo}")

            st.subheader("Roadmap Semanal de Oportunidade")
            icones_ceu = {'Ensolarado': '☀️', 'Nuvens Esparsas': '⛅', 'Nublado': '☁️', 'Chuva Leve': '🌦️',
                          'Chuva Forte': '🌧️'}
            cores_farol_hex = {"VERDE": "#28a745", "AMARELO": "#ffc107", "VERMELHO": "#dc3545"}
            num_dias = len(df_previsao);
            cols_roadmap = st.columns(num_dias)
            for i in range(num_dias):
                with cols_roadmap[i]:
                    dia = df_previsao.iloc[i];
                    cor = cores_farol_hex.get(dia['Farol']);
                    data_obj = pd.to_datetime(dia['data']);
                    dia_semana_curto = format_date(data_obj, "E", locale='pt_BR').capitalize();
                    dia_mes = data_obj.strftime('%d/%m')
                    st.markdown(f"""
                    <div style="background-color: {cor}; border-radius: 10px; padding: 10px; text-align: center; color: white; height: 175px; display: flex; flex-direction: column; justify-content: space-around;">
                        <div style="font-weight: bold; font-size: 1.1em;">{dia_semana_curto}</div><div style="font-size: 0.9em;">{dia_mes}</div>
                        <div style="font-size: 2.5em; margin: 5px 0;">{icones_ceu.get(dia['tipo_ceu'], '❓')}</div>
                        <div style="font-size: 0.8em; line-height: 1.2;">🌡️ {dia['temperatura_max_c']:.0f}°C<br>💧 {dia['precipitacao_diaria_mm']:.0f}mm</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

            col_mapa, col_detalhes = st.columns([2, 3])
            with col_mapa:
                st.subheader("Localização da Análise")
                map_center = [coords['lat'], coords['lon']]

                m = folium.Map(
                    location=map_center,
                    zoom_start=14,
                    tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                    attr='Google'
                )

                folium.Circle(
                    location=map_center,
                    radius=RAIO_PREVISAO_KM * 1000,
                    color="#ffffff",
                    weight=2,
                    fill=True,
                    fill_color="#3186cc",
                    fill_opacity=0.2,
                    tooltip=f"Área de validade da previsão ({RAIO_PREVISAO_KM} km)"
                ).add_to(m)

                folium.Marker(
                    location=map_center,
                    popup=f"<strong>{LOCAIS_PREDETERMINADOS[local_selecionado]['nome_curto']}</strong>",
                    tooltip=local_selecionado,
                    icon=folium.Icon(color='red', icon='industry', prefix='fa')
                ).add_to(m)

                st_folium(m, width='100%', height=400, returned_objects=[])

                st.caption(f"ℹ️ O círculo indica a área de validade da previsão (raio de {RAIO_PREVISAO_KM} km).")

            with col_detalhes:
                st.subheader("Análise Detalhada e Recomendações")
                for index, row in df_previsao.iterrows():
                    emoji_farol = {"VERDE": "🟢", "AMARELO": "🟡", "VERMELHO": "🔴"};
                    data_obj = pd.to_datetime(row['data']);
                    dia_semana = format_date(data_obj, "EEEE, dd/MM", locale='pt_BR').capitalize()
                    with st.expander(f"{emoji_farol.get(row['Farol'])} **{dia_semana}** - Farol: **{row['Farol']}**"):
                        st.markdown(f"**Insight:** {row['Insight']}");
                        st.info(f"**Dica Acionável:** {row['Dica']}")
                        st.markdown(
                            f"**Temp. Máx:** {row['temperatura_max_c']:.1f}°C &nbsp;&nbsp;|&nbsp;&nbsp; **Precipitação:** {row['precipitacao_diaria_mm']:.1f} mm &nbsp;&nbsp;|&nbsp;&nbsp; **Céu:** {row['tipo_ceu']}")
    else:
        st.error("Não foi possível obter a previsão do tempo.")
