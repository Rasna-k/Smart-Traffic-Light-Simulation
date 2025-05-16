import streamlit as st
from PIL import Image
import time
from ultralytics import YOLO
import io
import matplotlib.pyplot as plt
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Smart Traffic Light System", layout="wide")

# --- Heading ---
st.markdown(
    "<h1 style='color: #1E88E5;'>🚦 Smart Traffic Light System</h1>",
    unsafe_allow_html=True
)
st.markdown("Upload images from each direction of a 4-way intersection. The system will detect vehicles and assign green light duration based on vehicle count.")

# --- YOLOv8 Model ---
model = YOLO("yolov8n.pt")
vehicle_ids = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# --- Image Upload (4 columns) ---
st.markdown("### 📷 Upload Traffic Images")
cols = st.columns(4)
directions = ["North", "East", "South", "West"]
uploaded_images = {}

for col, direction in zip(cols, directions):
    with col:
        uploaded_images[direction] = st.file_uploader(f"{direction}", type=["jpg", "png"], key=direction)
        
# if st.button("🧹 Clear Uploads"):
#         # Clear session state related to uploads and downstream steps
#         for key in ["uploaded_files", "annotated_images", "counts", "sorted_directions", "current_index", "phase", "finished"]:
#             if key in st.session_state:
#                 del st.session_state[key]
#         st.rerun()

# --- Process Once All Images Are Uploaded ---
if all(uploaded_images.values()):
    # Initialize session state attributes if not already initialized
    if "annotated_images" not in st.session_state:
        st.session_state.annotated_images = {}
        st.session_state.counts = {}
        st.session_state.finished = set()  # Initialize the 'finished' attribute

        for direction, img_file in uploaded_images.items():
            img = Image.open(img_file).convert("RGB")
            results = model(img)

            # Count vehicles
            count = sum(1 for c in results[0].boxes.cls if int(c.item()) in vehicle_ids)
            st.session_state.counts[direction] = count

            # Save annotated image to memory
            annotated_array = results[0].plot()
            annotated_img = Image.fromarray(annotated_array[..., ::-1])  # Convert BGR → RGB
            st.session_state.annotated_images[direction] = annotated_img

        # Sort directions by vehicle count (descending)
        st.session_state.sorted_directions = sorted(st.session_state.counts.items(), key=lambda x: x[1], reverse=True)
        st.session_state.current_index = 0
        st.session_state.phase = "green"

    current_direction, current_count = st.session_state.sorted_directions[st.session_state.current_index]

    # --- Dynamic Green Time Logic ---
    base_time = 5
    time_per_vehicle = 1
    max_time = 25
    green_time = min(base_time + int(current_count/2) * time_per_vehicle, max_time)
    yellow_time = 3  # Fixed yellow duration

    # --- Enhanced Signal Status Visualization ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚥 Current Signal Status")

    signal_cols = st.columns(4)
    for idx, direction in enumerate(directions):
        with signal_cols[idx]:
            count = st.session_state.counts[direction]

            # Define color and message for current signal
            if direction == current_direction:
                if st.session_state.phase == "green":
                    color = "#4CAF50"  # Green
                    status = "🟢 GREEN"
                else:
                    color = "#FFC107"  # Yellow
                    status = "🟡 YELLOW"
            else:
                color = "#D32F2F"  # Red
                status = "🔴 RED"

            # Highlight only if green or yellow
            style = f"""
                <div style='
                    background-color:{color};
                    padding:20px;
                    border-radius:10px;
                    color:white;
                    font-size:24px;
                    font-weight:bold;
                    box-shadow:0 0 10px rgba(0,0,0,0.3);'>
                    {direction}<br>{status}<br>{count} Vehicles
                </div>
            """
            if direction == current_direction:
                st.markdown(style, unsafe_allow_html=True)
            else:
                st.markdown(f"### {status}\n**{direction}**\n**{count} Vehicles**")

    # --- Timer UI ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⏱ Signal Timer")
    timer_placeholder = st.empty()

    # --- Display Annotated Images Below Timer ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🖼️ Detected Vehicles (Annotated Images)")
    img_cols = st.columns(4)
    for idx, direction in enumerate(directions):
        with img_cols[idx]:
            st.image(st.session_state.annotated_images[direction], caption=f"{direction} - Vehicles: {st.session_state.counts[direction]}", use_container_width=True)

    # --- Countdown Timer ---
    duration = green_time if st.session_state.phase == "green" else yellow_time
    for remaining in range(duration, 0, -1):
        if st.session_state.phase == "green":
            timer_placeholder.markdown(f"🟢 Green light for **{current_direction}** - {remaining} sec")
        else:
            timer_placeholder.markdown(f"🟡 Yellow light for **{current_direction}** - {remaining} sec")

        time.sleep(1)

    # --- Switch Phase or Stop ---
    if st.session_state.phase == "green":
        st.session_state.phase = "yellow"
        st.rerun()
    else:
        # After yellow, mark direction as finished
        st.session_state.finished.add(current_direction)
        st.session_state.current_index += 1
        st.session_state.phase = "green"

        # Stop rerun if all directions are finished
        if len(st.session_state.finished) < 4:
            st.rerun()
        else:
            # Show completion message when all directions are done
            st.success("### 🚦 Simulation Complete! All directions have completed their green and yellow phases.")
            counts = st.session_state.counts
            sorted_directions = st.session_state.sorted_directions
            # st.session_state.clear()
            # Add restart button
            # st.markdown("<br>", unsafe_allow_html=True)
            # if st.button("🔄 Restart Simulation"):
                # for key in [
                #     "annotated_images", "counts", "sorted_directions",
                #     "current_index", "phase", "finished"
                # ]:
                #     if key in st.session_state:
                #         del st.session_state[key]
                #    st.markdown(
                #                 """
                #                 <script>
                #                     window.location.reload();
                #                 </script>
                #                 """,
                #                 unsafe_allow_html=True
                #             )
            
            st.markdown("<br>", unsafe_allow_html=True)
            # st.markdown("Click below to view detailed traffic insights 👇")

            # if st.button("📊 Show Dashboard"):    
                # --- Dashboard ---
            
                
            st.markdown("## 📊 Traffic Analysis Dashboard")

            # 1. Total vehicles
            total_vehicles = sum(st.session_state.counts.values())                
            st.markdown(
                            f"""
                            <br>
                            <h3 style='font-size: 24px; font-weight: bold;'>🚗 Total Vehicles Detected: <span style='color: #4da6ff;'>{total_vehicles}</span></h3>""",unsafe_allow_html=True)
                
            # 2. Bar chart for vehicle count
            st.markdown("<br><h3 style='font-size: 24px; font-weight: bold;'>🚦 Vehicle Count per Direction</h3>", unsafe_allow_html=True)
            # Create the bar chart
            vehicle_counts = st.session_state.counts                
            fig, ax = plt.subplots(figsize=(4,4))

            # Plot the bar char
            ax.bar(vehicle_counts.keys(), vehicle_counts.values(), color='teal')

            # Add numbers on top of the bars
            for i, count in enumerate(vehicle_counts.values()):                  
                ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold', fontsize=8)

            # Customize the chart
            ax.set_ylabel('Vehicle Count')
            ax.set_xlabel('Directions')
            ax.set_title('Vehicle Count per Direction')           
            # Display the chart in Streamlit
            # Show in small column
            col1, col2, _ = st.columns([.3, .4, .3])
            with col2:
                st.pyplot(fig)

            # 3. Most congested direction
            busiest = max(st.session_state.counts.items(), key=lambda x: x[1])
            st.markdown(
                            f"""
                            <br>
                            <h3 style='font-size: 24px; font-weight: bold;'>
                                🥇 Busiest Direction: <span style='color: red;'>{busiest[0]}</span> with {busiest[1]} vehicles
                            </h3>
                            """,
                            unsafe_allow_html=True
                        )

                # 4. Green time per direction
            base_time = 5
            time_per_vehicle = 1
            max_time = 25                
            # Use the same formula applied earlier
            green_times = [
                    min(base_time + int(st.session_state.counts[d] / 2) * time_per_vehicle, max_time)
                    for d, _ in st.session_state.sorted_directions
                ]
            green_df = pd.DataFrame({
                            "Direction": [d for d, _ in sorted_directions],
                            "Assigned Green Time (sec)": green_times
                })
            st.markdown("<br><h3 style='font-size: 24px; font-weight: bold;'>⏱️ Assigned Green Time</h3>" , unsafe_allow_html=True)
            col1, col2, _ = st.columns([.3, .4, .3])
            with col2:
                    st.dataframe(green_df.style.set_properties(**{
                            'background-color': '#e8f5e9',
                            'color': 'green',
                            'font-size': '12px',
                            'font-weight': 'bold',
                            'text-align' : 'left'
                            }), use_container_width=True)

            # 5. Pie chart (optional)
            import pandas as pd
            df = pd.DataFrame.from_dict(st.session_state.counts, orient='index', columns=['Vehicles'])
            st.markdown("<h4 style='font-size: 20px; font-weight: bold;'>📈 Traffic Share by Direction</h4>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(4, 4))  # Smaller size
            ax2.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', textprops={'fontsize': 8})               
            ax2.set_title("Traffic Distribution", fontsize=10)
            col1, col2, _ = st.columns([.2, .4, .2])
            with col2:
                    st.pyplot(fig2)
                    
            # 6. Waiting Time Calculation ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### ⏱️ Waiting Time Analysis")
            # Use the already computed busiest direction
            busiest_direction = busiest[0]

            # Define green time per vehicle
            time_per_vehicle = 1
            # Initialize dictionary to store waiting times
            waiting_times = {}

            # Calculate waiting times for all directions except the busiest one
            for direction in directions:
                    if direction == busiest_direction:
                        waiting_times[direction] = 0  # No wait for the first green
                    else:
                        # Waiting time is the sum of green times of all directions before this one (excluding busiest and current)
                        preceding_directions = [d for d, _ in st.session_state.sorted_directions if d != direction and d != busiest_direction]
                        wait = sum(min(5 + st.session_state.counts[d] * time_per_vehicle, 30) + 3  # green + yellow
                                for d in preceding_directions)
                        waiting_times[direction] = wait

                # Display waiting times for each direction
            waiting_time_data = pd.DataFrame({
                    "Direction": directions,
                    "Waiting Time (sec)": [waiting_times[direction] for direction in directions]
            })

            # Display waiting times in a table
            # Show in small column
            col1, col2, _ = st.columns([.3, .4, .3])
            with col2:
                    st.dataframe(waiting_time_data.style.set_properties(**{
                    'background-color': '#f3f4f6',
                    'color': 'black',
                    'font-size': '12px',
                    'font-weight': 'bold'
                    }))

            # Optionally, add a message for the waiting times
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                    """
                    The waiting times represent the total waiting time for vehicles at each direction based on the vehicle counts in other directions.
                    The higher the vehicle count in other directions, the higher the waiting time for the current direction.
                    """
                )
            st.session_state.clear()