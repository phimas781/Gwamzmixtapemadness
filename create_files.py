# create_files.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Create the dataset
def create_gwamz_data():
    data = {
        'artist_name': ['Gwamz']*19,
        'artist_id': ['5asNjXXCZBu2NfJDarWlRR']*19,
        'artist_followers': [7937]*19,
        'artist_popularity': [41]*19,
        'album_name': [
            "Bad to The Bone (feat. Gwamz, Odzmoney, Madison B & Jedz) [New Gen Remix]",
            "Last Night (Sped Up)", "Last Night (Sped Up)", "Last Night (Sped Up)",
            "Just2", "Just2", "PAMELA", "French Tips", "French Tips",
            "C'est La Vie", "Last Night (Jersey Edit)", "Last Night (Jersey Edit)",
            "Last Night (Jersey Edit)", "Composure", "Like This", "Like This",
            "Just2 (Jersey Club)", "Just2 (Jersey Club)", "Just2 (Jersey Club)",
            "Last Night"
        ],
        'album_id': [
            '7s9XSPSca3DeXsmzA97LUX', '7fT2dJ4unWRH5OPVoB2x7T', '7fT2dJ4unWRH5OPVoB2x7T',
            '7fT2dJ4unWRH5OPVoB2x7T', '12VZgPe5LwxXECl7XdHCMl', '12VZgPe5LwxXECl7XdHCMl',
            '3sgshiWcl1YVV4ywA1GEp2', '3QR5pcsLpaXWRPMavi8jIg', '3QR5pcsLpaXWRPMavi8jIg',
            '3a9e6LG7hv9LuJFQQ6YOxW', '5NsYFfSbxdPtkBebw7CkiQ', '5NsYFfSbxdPtkBebw7CkiQ',
            '5NsYFfSbxdPtkBebw7CkiQ', '4Fe2qUD57mqhbn2gPQrlpa', '4BbE5HbCi5gtek3iVPK4Vr',
            '4BbE5HbCi5gtek3iVPK4Vr', '4CdrS1C25z43W7d3dYXk5U', '4CdrS1C25z43W7d3dYXk5U',
            '4CdrS1C25z43W7d3dYXk5U', '5S3zsQMIcfVbg5zMW3mn1E'
        ],
        'album_type': ['single']*19,
        'release_date': [
            '07/03/2025', '16/03/2023', '16/03/2023', '16/03/2023', '18/04/2024',
            '18/04/2024', '12/10/2023', '30/05/2025', '30/05/2025', '29/04/2021',
            '26/05/2023', '26/05/2023', '26/05/2023', '11/02/2022', '22/11/2024',
            '22/11/2024', '19/09/2024', '19/09/2024', '19/09/2024', '09/03/2023'
        ],
        'release_year': [2025, 2023, 2023, 2023, 2024, 2024, 2023, 2025, 2025, 
                         2021, 2023, 2023, 2023, 2022, 2024, 2024, 2024, 2024, 2024, 2023],
        'total_tracks_in_album': [1, 3, 3, 3, 2, 2, 1, 2, 2, 1, 3, 3, 3, 1, 2, 2, 3, 3, 3, 1],
        'available_markets_count': [185]*19,
        'track_name': [
            "Bad to The Bone (feat. Gwamz, Odzmoney, Madison B & Jedz) - New Gen Remix",
            "Last Night - Sped Up", "Last Night", "Last Night - Instrumental",
            "Just2 - Sped Up", "Just2", "PAMELA", "French Tips - Sped Up", "French Tips",
            "C'est La Vie", "Last Night - Jersey Edit", "Last Night", "Last Night - Sped Up",
            "Composure", "Like This", "Like This - Sped Up", "Just2 - Sped Up", "Just2",
            "Just2 - Jersey Club", "Last Night"
        ],
        'track_id': [
            '5zlvL9Uc16QXrpEIRr5a5y', '5wyc7kwPQYWQ9NFQDLaEJu', '5KPqtyUqDj6IEE75CsRcEs',
            '2UG2ML4GEfOEFfoMT1sFVU', '4Petb0rQa8YFvxe1TInhUP', '7b56MrXXeTox20pq5nLroD',
            '4G3axOEaeRZ95EIbubQbQu', '7uhyKcf4NMViOEMaTljUNX', '3UattpReJNxvgzDkI0eIa0',
            '2t7siSqkXNEjtwxYWsK6yh', '5qCFd6E1JEGqwpBEIAYnH0', '09rpee1dVXXWFNTpE3ZBUO',
            '470UZ36MTbVvqAuWlWWF4q', '41DWlrwHckj4Mq2XBhrYE8', '3rrb5kb3UrpIsU8yqMx1B5',
            '5eiMHDYi09A0Rd4rGGyt9l', '0Emd7kqZ724laj8WPksz2b', '40IjZHh682LHCjEf1GhWyg',
            '6d2qhM4lVurSRYWCuBsTGC', '1yfRzQ5wzYRFZk1WHtOqtQ'
        ],
        'track_number': [1, 1, 2, 3, 1, 2, 1, 1, 2, 1, 1, 2, 3, 1, 1, 2, 1, 2, 3, 1],
        'disc_number': [1]*19,
        'explicit': [True, True, True, False, True, True, True, False, False, True, 
                     True, True, True, True, True, True, True, True, True, True],
        'track_popularity': [38, 26, 16, 4, 45, 43, 39, 41, 41, 22, 4, 9, 7, 25, 51, 36, 24, 12, 24, 44],
        'Number Of Strems': [
            127317, 308914, 2951075, 8637, 2149598, 1157082, 766818, 66852, 65482, 
            56899, 8473, 2951075, 308914, 81110, 1724835, 200204, 2149598, 1157082, 64730, 1724835
        ]
    }
    return pd.DataFrame(data)

# Create the predictive model
def create_gwamz_model(df):
    # Feature engineering
    df['version_type'] = df['track_name'].apply(
        lambda x: 'sped_up' if 'sped up' in x.lower() 
        else 'remix' if 'remix' in x.lower() 
        else 'instrumental' if 'instrumental' in x.lower() 
        else 'edit' if 'edit' in x.lower() 
        else 'jersey_club' if 'jersey' in x.lower() 
        else 'original'
    )
    df['release_date'] = pd.to_datetime(df['release_date'], format='%d/%m/%Y')
    df['release_month'] = df['release_date'].dt.month
    df['explicit'] = df['explicit'].astype(int)
    df['days_since_first_release'] = (df['release_date'] - df.groupby('track_name')['release_date'].transform('min')).dt.days
    
    # One-hot encoding
    version_dummies = pd.get_dummies(df['version_type'], prefix='version')
    df = pd.concat([df, version_dummies], axis=1)
    
    # Select features
    features = ['release_year', 'release_month', 'total_tracks_in_album', 
                'explicit', 'days_since_first_release', 
                'version_original', 'version_sped_up', 'version_remix',
                'version_edit', 'version_jersey_club', 'version_instrumental']
    
    # Create target
    df['streams'] = df['Number Of Strems']
    
    # Train model
    X = df[features]
    y = df['streams']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Create and save files
df = create_gwamz_data()
model = create_gwamz_model(df)

df.to_csv('gwamz_data.csv', index=False)
joblib.dump(model, 'gwamz_model.pkl')

print("Files created successfully!")
print("gwamz_data.csv and gwamz_model.pkl are ready in your current directory")
