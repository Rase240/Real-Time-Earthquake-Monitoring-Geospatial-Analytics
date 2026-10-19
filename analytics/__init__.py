"""
TerraWatcher analytical modules.

  geo                 great-circle geometry shared by every module
  catalog             validated event tables and the historical catalogue download
  regions             the fixed 10-degree region grid and weekly count matrices
  regional_features   forecast-origin features (one definition for training and serving)
  sequences           space-time DBSCAN sequence detection (Module A)
  forecast            seven-day regional count forecasts (Module B)
  anomaly             elevated-activity detection against a historical reference (Module C)
  aftershock          event-centred sequence statistics and Omori-Utsu fitting (Module D)
"""
