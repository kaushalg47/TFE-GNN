import os
import subprocess

# Paths
pcap_folder = ""  
output_base_folder = ""  

# Create output base folder if it doesn't exist
os.makedirs(output_base_folder, exist_ok=True)

# Define categories and their corresponding ports
categories = {
    "chat": [443,80,40011,47872,40273,12350,40019],  # Example ports for chat applications
    "email": [25, 110, 143, 993, 995],  # SMTP, POP3, IMAP
    "file": [20, 21, 989, 990],  # FTP
    "p2p": [6881, 6889],  # BitTorrent
    "streaming": [1935, 554, 1755],  # RTMP, RTSP, Microsoft Media Server
    "VoIP": [5060, 5061]  # SIP
}

# Iterate through all pcap files in the folder
for file in os.listdir(pcap_folder):
    if file.endswith(".pcap"):
        input_pcap = os.path.join(pcap_folder, file)
        
        # Process each category
        for category, ports in categories.items():
            # Create a unique output directory for each category
            category_folder = os.path.join(output_base_folder, category)
            os.makedirs(category_folder, exist_ok=True)
            
            # Construct the SplitCap command for each category
            ports_str = ' '.join([f"-port {port}" for port in ports])
            splitcap_command = f"SplitCap.exe -r {input_pcap} -s session {ports_str} -o {category_folder}"
            print(f"Executing: {splitcap_command}")
            subprocess.run(splitcap_command, shell=True)

print("Processing complete.")
