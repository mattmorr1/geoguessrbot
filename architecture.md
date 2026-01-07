# geoguessrbot

**Architecture**

*Dataset*

- Download Openstreetview-5M dataset
- Use S2 Geometry Library for dividing map into cells
    - Place data based upon GPS associated data into cells
- Connect cluster with associated alphaearth embedding
- Build into vector DB

SCHEMA:
- item (id)
- gps
- location (box that it exists in)
- streetclip 768d vector
- pgvec/milvus (alphaearth embedding 64d)

*Model*

- Create static weighted version of streetclip
- Transform clip from 768d -> 64d (alphaearth)

*Inference*

- Cosine similarity search 64d vector with alphaearth in vectordb (milvus/pgvector)
- Top-k weighted interpolation function for prediction
- LIMIT 5 (pull top 3 most likely, % predicted)

*Testing*

- Enter Geocoords of orig alongside image (optional), also pull gps coords of alphaearth in db
- Haversine Formula of GPS coords (measure distance)

*UI*

- Ollama with image upload for user-friendly interaction
