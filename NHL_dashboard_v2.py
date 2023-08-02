#!/usr/bin/env python
# coding: utf-8

# In[172]:


import pandas as pd
import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape
import streamlit as st
import streamlit_authenticator as stauth
import pickle
from pathlib import Path
import requests
from lxml import etree
from lxml import html
from bs4 import BeautifulSoup
import json
import os
import shutil
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from collections import defaultdict
from urllib.parse import unquote
from urllib.parse import quote
from PIL import Image

import matplotlib.pyplot as plt
from PIL import ImageSequence
from PIL import ImageOps
import io


# In[173]:


# Load player information
player_info_df = pd.read_csv("Parameters.csv")
player_info_df = player_info_df.round(2)

# Load player stats
player_stats_df = pd.read_csv("Total_Points.csv")

# Team logos
def load_logo_csv():
    return pd.read_csv('nhl_team_logos.csv')

logo_df = load_logo_csv()


# In[174]:


# Function to convert time strings to decimal minutes
def time_to_decimal_minutes(time_str):
    if pd.notna(time_str):
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = 0
        if len(parts) > 2:
            seconds = int(parts[2])
        return hours + minutes / 60 + seconds / 3600
    else:
        return 0

# Convert "TOI (min)" column to decimal minutes
player_info_df["TOI (min)"] = player_info_df["TOI (min)"].apply(time_to_decimal_minutes)

player_stats_df["Total TOI(min)"] = player_stats_df["Total TOI(min)"].apply(time_to_decimal_minutes)


# In[175]:


player_stats_df.rename(columns={"TOI/GP(min)": "TOI/GP"}, inplace=True)
player_info_df.rename(columns={"Expected Goals For WOI": "xGF WOI", "Expected Goals Against WOI": "xGA WOI"}, inplace=True)


# In[176]:


# User Auth
names = ["Sean Farquharson", "Michael Perelman"]
usernames = ["sfarg", "mperelman"]

file_path = Path("/Users/SFarquharson/Documents/Blues_project") / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)
    
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                   "nhllines_dashboard","abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")
    
if authentication_status == None:
    st.warning("Please enter your username and password")
    
if authentication_status:

    def parsePlayerTable(tree):
        all_players_xpath_str = "//div[contains(@class, 'w-1/3') or contains(@class, 'w-1/2')]//a/span[@class='text-xs font-bold uppercase xl:text-base']"
        all_images_xpath_str = "//div[contains(@class, 'w-1/3') or contains(@class, 'w-1/2')]//img[contains(@srcset, 'https')]"

        all_players_elements = tree.xpath(all_players_xpath_str)
        all_images_elements = tree.xpath(all_images_xpath_str)

        all_players = [elem.text.strip() for elem in all_players_elements]
        all_images = [unquote(elem.attrib['srcset']).split(', ')[-1].split()[0] for elem in all_images_elements] # Extract image URLs and decode them

        # split players into forward lines and defensive pairings
        forwards = all_players[:12]
        forward_images = all_images[:12]
        defense = all_players[12:18]
        defense_images = all_images[12:18]

        return forwards, forward_images, defense, defense_images

    def fuzzy_match(name, list_names, min_score=0):
        # Returns a tuple of the best match along with its similarity score, 
        # but only if it's above a defined threshold.
        max_score = -1
        max_name = ""
        for name2 in list_names:
            score = fuzz.ratio(name, name2)
            if (score > min_score) & (score > max_score):
                max_name = name2
                max_score = score
        return (max_name, max_score)

    # WORKING **********************************************************************************************

    def fetch_team_lines(team, all_players, player_info_df):
        url = f"https://www.dailyfaceoff.com/teams/{team.lower().replace(' ', '-')}/line-combinations/"
        base_url = "https://www.dailyfaceoff.com"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for 4xx and 5xx errors
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data: {e}")
            return []

        tree = etree.HTML(response.text)
        forwards, forward_image_urls, defense, defense_image_urls = parsePlayerTable(tree)

        # prepend base_url to each image url
        forward_image_urls = [base_url + url for url in forward_image_urls]
        defense_image_urls = [base_url + url for url in defense_image_urls]

        matched_players = defaultdict(list)
        player_image_urls = defaultdict(list)

        # match forwards
        for player_name, image_url in zip(forwards, forward_image_urls):
            match, score = fuzzy_match(player_name, all_players, min_score=70)
            if match and player_info_df[player_info_df['Player'] == match]['TOI (min)'].iloc[0] >= 300:
                matched_players['forwards'].append(match)
                player_image_urls['forwards'].append(image_url)
            elif match:
                matched_players['forwards'].append(match + ' ❌')
                player_image_urls['forwards'].append(image_url)

        # match defense
        for player_name, image_url in zip(defense, defense_image_urls):
            match, score = fuzzy_match(player_name, all_players, min_score=70)
            if match and player_info_df[player_info_df['Player'] == match]['TOI (min)'].iloc[0] >= 300:
                matched_players['defense'].append(match)
                player_image_urls['defense'].append(image_url)
            elif match:
                matched_players['defense'].append(match + ' ❌')
                player_image_urls['defense'].append(image_url)


        return matched_players, player_image_urls

    def main():
        st.markdown("<h1 style='text-align: center; color: black;'><i><b>NHL LINES ADVANCED</b></i></h1>", unsafe_allow_html=True)

        st.markdown("<h6 style='text-align: center; color: black;'>By Sean Farquharson</h6>", unsafe_allow_html=True)

        scope_expander = st.expander("Show General Scope")
        with scope_expander:
            st.markdown("""
                Welcome to the NHL lines advanced dashboard. This tool is designed to provide an in-depth analysis of NHL team lines using advanced stats. Each player's performance is analyzed in comparison to league totals and averages, providing a comprehensive overview of their capabilities. 

                In the dashboard, players' names may be accompanied by various icons, the meaning of which can be found in the legend below. An icon next to a player's name indicates that they are above average in the associated category. 

                You have the option to navigate through player profiles or an overall leaderboard showcasing the stats included in our analysis. Please note that players with less than 300 total minutes played are marked with an 'x' emoji. Although I don't provide detailed stats for these players, I include them in the line-up view for completeness. 

                This dashboard satisfies the minimum requirements of the project and offers an innovative tool that can be immediately put to use by an NHL organization. I hope you find it useful and informative.
            """)
        
        authenticator.logout("Logout", "sidebar")
        st.sidebar.title(f"Welcome {name}")
        st.sidebar.header('Analyze Team')

        # Sidebar for team selection
        selected_team = st.sidebar.selectbox("Select NHL Team", player_info_df["Team"].unique(), index=player_info_df["Team"].unique().tolist().index("St Louis Blues"))

        # Sidebar - Navigation
        st.sidebar.header('Navigation')
        page = st.sidebar.selectbox('Select Page', ['Player Profile', 'League Leaderboard'])

        # Load logo for the selected team
        logo_url = logo_df[logo_df['Team'] == selected_team]['Logo'].values[0]

        # Display the team's logo below the title, centered
        st.markdown(f"<center><img src='{logo_url}'></center><br>", unsafe_allow_html=True)

        # Convert selected_team to the correct format (lowercase with hyphens)
        formatted_team = selected_team.lower().replace(' ', '-')

        # Extract all player names
        all_players = player_info_df['Player'].unique().tolist()

        # Fetch team lines and player image urls from dailyfaceoff.com
        team_lines, player_image_urls = fetch_team_lines(formatted_team, all_players, player_info_df)

        st.markdown("<div style='background-color: #F0F0F0; padding: 10px; border: 1px solid black; border-radius: 5px;'><b><span style='color: red;'>❌</span> Player has less than 300 minutes played</b><br><b><span style='color: blue;'>🛡️</span> Player has more than 16 successful defensive touches (5v5 per 60mins)</b><br><b><span style='color: green;'>🟢</span> Player's xGF WOI is greater than xGA WOI (5v5 per 60mins)</b><br><b><span style='color: red;'>🔴</span> Player's xGF WOI is less than xGA WOI (5v5 per 60mins)</b><br><b><span style='color: brown;'>🏒</span> Player has greater than or equal to 43 Possession Driving Plays (5v5 per 60mins)</b><br><b><span style='color: yellow;'>⚠️</span> Player has more than 4.04 Total Shot From Slot Attempts (5v5 per 60mins)</b></div>", unsafe_allow_html=True)

        # Display team lines
        team_players = []
        for position_group, players in team_lines.items():
            st.markdown(f"<h2 style='text-align: center; color: black;'><b>{position_group.upper()}</b></h2>", unsafe_allow_html=True)
            if position_group == 'forwards':
                grouping = 3
            else:
                grouping = 2
            grouped_players = [players[n:n+grouping] for n in range(0, len(players), grouping)]
            image_urls_grouped = [player_image_urls[position_group][n:n+grouping] for n in range(0, len(players), grouping)]

            for group, image_urls in zip(grouped_players, image_urls_grouped):
                cols = st.columns([1]*(grouping - len(group)) + [2]*len(group) + [1]*(grouping - len(group)))

                # Initialize the differential counter for this line
                line_differential = 0

                for i, (player_name, image_url) in enumerate(zip(group, image_urls)):
                    player_info = player_info_df[player_info_df['Player'] == player_name]
                    if not player_info.empty:
                        defensive_touches = player_info['Successful Defensive Touches'].values[0]
                        xGF_WOI = player_info['xGF WOI'].values[0]
                        xGA_WOI = player_info['xGA WOI'].values[0]
                        possession_driving_plays = player_info['Possession Driving Plays'].values[0]
                        total_shot_from_slot_attempts = player_info['Total Shot From Slot Attempts'].values[0]

                        # Update the line differential
                        line_differential += (xGF_WOI - xGA_WOI)

                        player_name_display = player_name
                        if defensive_touches > 16:
                            player_name_display += ' 🛡️'
                        if xGF_WOI > xGA_WOI:
                            player_name_display += ' 🟢'
                        elif xGF_WOI < xGA_WOI:
                            player_name_display += ' 🔴'
                        if possession_driving_plays >= 43:
                            player_name_display += ' 🏒'
                        if total_shot_from_slot_attempts > 4.04:
                            player_name_display += ' ⚠️'
                    else:
                        player_name_display = player_name
                    cols[i+(grouping - len(group))].markdown(f"<center><img src='{image_url}' width='90'></center>", unsafe_allow_html=True)
                    cols[i+(grouping - len(group))].markdown(f"<center>{player_name_display}</center>", unsafe_allow_html=True)
                    team_players.append(player_name)

                # Display the line differential after each line
                st.markdown(f"<h6 style='text-align: center; color: black;'><b>xG Differential: {round(line_differential, 2)}</b></h6>", unsafe_allow_html=True)

        if page == 'Player Profile':

            # Player profile section
            st.sidebar.markdown("<h2 style='text-align: center; color: black;'><i><b>PLAYER PROFILE</b></i></h2>", unsafe_allow_html=True)

            # Dropdown to select player for player profile
            selected_player = st.sidebar.selectbox("Select Player", team_players)

            # Filter player stats based on selected player
            player_profile_stats = player_stats_df[player_stats_df['Player'] == selected_player][['Position', 'GP', 'TOI/GP', 'G', 'A', 'PTS', '+/-', 'S']]
            player_info = player_info_df[player_info_df['Player'] == selected_player][['xGF WOI', 'xGA WOI', 'Total Shot From Slot Attempts', 'Possession Driving Plays', 'Successful Defensive Touches']]

            # Display player profile stats
            st.sidebar.markdown("<h3 style='text-align: center; color: black;'><i><b>PLAYER STATS</b></i></h3>", unsafe_allow_html=True)
            for i in range(0, len(player_profile_stats.columns), 2):
                cols = st.sidebar.columns(2)
                for j in range(2):
                    if i + j < len(player_profile_stats.columns):
                        column = player_profile_stats.columns[i + j]
                        value = player_profile_stats[column].values[0]
                        cols[j].markdown(f"<b>{column}</b> {value}", unsafe_allow_html=True)

             # Displaying title for advanced stats
            st.sidebar.markdown("<h3 style='text-align: center; color: black;'><i><b>ADVANCED STATS</b></i><br><i style='font-size: 15px;'>(5v5 per 60mins)</i></h3>", unsafe_allow_html=True)

            # Display player info
            for i in range(0, len(player_info.columns), 2):
                cols = st.sidebar.columns(2)
                for j in range(2):
                    if i + j < len(player_info.columns):
                        column = player_info.columns[i + j]
                        value = player_info[column].values[0]
                        cols[j].markdown(f"<b>{column}</b> {value}", unsafe_allow_html=True)

            # Add Histograms for selected stats here
            st.sidebar.markdown("<h3 style='text-align: center; color: black;'><i><b>HISTOGRAMS</b></i></h3>", unsafe_allow_html=True)

            plt.style.use('ggplot')  # Apply a style

            # Override grid settings
            plt.rc('axes', grid=False)
            plt.rc('axes', facecolor='white')

            for stat in ['xGF WOI', 'xGA WOI', 'Total Shot From Slot Attempts', 'Possession Driving Plays', 'Successful Defensive Touches']:
                selected_player_stat = player_info_df[player_info_df['Player'] == selected_player][stat].values[0]
                average_stat = player_info_df[player_info_df['TOI (min)'] > 300][stat].mean()

                all_players_stat = player_info_df[player_info_df['TOI (min)'] > 300][stat]

                # Create the bins
                bins = np.linspace(all_players_stat.min(), all_players_stat.max(), 30)

                # Create a figure and axis object
                fig, ax = plt.subplots()

                # Load the gif from the URL
                response = requests.get(logo_url)
                gif = Image.open(io.BytesIO(response.content))

                # Convert gif to png
                pngs = []
                for frame in ImageSequence.Iterator(gif):
                    png = frame.convert('RGBA')
                    pngs.append(png)
                img = pngs[0]  # Use only first frame of gif

                # Add a white border around the image to effectively "shrink" its size within the plot
                border_size = int(max(img.size) * 99 / 100)  
                img_with_border = ImageOps.expand(img, border=border_size, fill='white')

                # Convert PIL Image to numpy array
                img = np.asarray(img_with_border).copy()  # Make a copy of the array

                # Fade out the image by adjusting the alpha channel
                img[..., -1] = img[..., -1] * 0.1

                # Plot the image in the background, set zorder to -1 so it's behind other plot elements
                ax.imshow(img, extent=[all_players_stat.min(), all_players_stat.max(), 0, 120], zorder=-1, aspect='auto')

                # Create histogram for each bar
                for b in bins[:-1]:
                    # Select data for this bar
                    bar_data = all_players_stat[(all_players_stat >= b) & (all_players_stat < b + bins[1] - bins[0])]
                    color = 'skyblue' if (bar_data.mean() < selected_player_stat) else 'lightgrey'
                    ax.hist(bar_data, bins=[b, b + bins[1] - bins[0]], color=color, edgecolor='black', alpha=0.5)

                # Add vertical line for the selected player's stat
                ax.axvline(selected_player_stat, color='r', linestyle='dashed', linewidth=2, label=selected_player)

                # Add a title and labels
                ax.set_title(f'{stat} Distribution', fontsize=14)
                ax.set_xlabel(stat, fontsize=12)
                ax.set_ylabel('Frequency', fontsize=12)

                # Enable grid
                ax.grid(True)

                # Set y limit to fix the scaling issue
                ax.set_ylim([0, 80])  # Set the maximum limit to 100

                # Adjust the layout
                fig.tight_layout()

                # Add legend
                ax.legend()

                st.sidebar.pyplot(fig)  # Pass figure to st.pyplot()
                plt.clf()  # Clear the current figure after plotting

            # Show message if no stats available
            if player_profile_stats.empty and player_info.empty:
                st.sidebar.write(f"No stats available for {selected_player}")

        elif page == 'League Leaderboard':
            st.sidebar.header("League Leaderboard")

            # Filter player_stats_df for players with more than 300 minutes played
            filtered_player_stats_df = player_stats_df[player_stats_df['Total TOI(min)'] > 300]

            # Filter player_info_df for players with more than 300 minutes played
            filtered_player_info_df = player_info_df[player_info_df['TOI (min)'] > 300]

            # List of stats to display
            stats_to_display = ['G', 'A', 'PTS', '+/-', 'S', 'xGF WOI', 'xGA WOI', 'Total Shot From Slot Attempts', 'Possession Driving Plays', 'Successful Defensive Touches']

            # Allow user to select stat
            selected_stat = st.sidebar.selectbox('Select Stat', stats_to_display)

            if selected_stat in ['G', 'A', 'PTS', '+/-', 'S']:
                # Display top 10 players for selected stat from player_stats_df
                top_players = filtered_player_stats_df.nlargest(10, selected_stat)[['Player', selected_stat]]
                st.sidebar.markdown(f"#### Top 10 Players for {selected_stat}")
                for i, row in top_players.iterrows():
                    st.sidebar.markdown(f"{row['Player']}: {row[selected_stat]}")

            else:
                # Display top 10 players for selected stat from player_info_df
                top_players_info = filtered_player_info_df.nlargest(10, selected_stat)[['Player', 'Team', selected_stat]]
                st.sidebar.markdown(f"#### Top 10 Players for {selected_stat}")
                for i, row in top_players_info.iterrows():
                    st.sidebar.markdown(f"{row['Player']} ({row['Team']}): {row[selected_stat]}")

    if __name__ == "__main__":
        main()


# In[ ]:




