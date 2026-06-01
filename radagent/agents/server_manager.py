from typing import Dict, List, Any
from fastmcp import Client
import logging
import sys
import os
import select
import asyncio
import time
import threading
import queue
import subprocess
import aiohttp

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MultiServerManager:
    def __init__(self, servers: Dict[str, str] = None):
        self.servers = servers
        self.clients: Dict[str, Client] = {}
        self.tools: Dict[str, List] = {}

    async def connect_all(self):
        """Connect to all FastMCP servers and maintain connections"""
        for server_name in self.servers.keys():
            try:
                # Create FastMCP client and connect
                client = Client(self.servers[server_name]["addr"], timeout=3600 * 3)
                await client._connect()
                self.clients[server_name] = client

                # Get tool list from this server
                tools_result = await client.list_tools()
                self.tools[server_name] = tools_result
                logger.info(
                    f"Connected to {server_name} server with {len(tools_result)} tools"
                )

            except Exception as e:
                logger.error(f"Failed to connect to {server_name} server: {e}")
                raise

    async def get_all_tools(self) -> Dict[str, List[Dict]]:
        """Get information about all available tools"""
        all_tools = {}
        for server_name, tools in self.tools.items():
            all_tools[server_name] = []
            for tool in tools:
                # FastMCP tool objects may have different attributes, check available ones
                tool_info = {
                    "name": getattr(tool, "name", "unknown"),
                    "description": getattr(tool, "description", "") or "",
                    "input_schema": getattr(tool, "input_schema", {}),
                    "server": server_name,
                }
                all_tools[server_name].append(tool_info)
        return all_tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """Call a specific tool"""
        if server_name not in self.clients:
            raise ValueError(f"Server {server_name} not connected")

        try:
            client = self.clients[server_name]
            result = await client.call_tool(tool_name, arguments)
            return result.structured_content["outputs"]
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {server_name}: {e}")
            return None

    async def close_all(self):
        """Close all server connections"""
        for server_name, client in self.clients.items():
            try:
                await client._disconnect()
                logger.info(f"Disconnected from {server_name} server")
            except Exception as e:
                logger.error(f"Error disconnecting from {server_name}: {e}")
        self.clients.clear()

    async def startup_all_servers(self):
        processes = []
        output_queue = queue.Queue()
        output_threads = []
        t1 = time.time()

        # Start output monitor thread FIRST
        monitor_thread = threading.Thread(
            target=monitor_output_queue, args=(output_queue,), daemon=True
        )
        monitor_thread.start()
        logging.info("✅ Started output monitor thread")

        # start the server
        for server_config in self.servers.values():
            logging.info(
                f"Starting server: {server_config['script']} on port {server_config['port']}"
            )
            nodes_mapping = {
                i: ip for i, ip in enumerate(os.environ["NODES_IP"].split())
            }
            target_node = int(server_config.get("target_node", 0))
            target_node = target_node if target_node in nodes_mapping.keys() else 0
            target_node_ip = nodes_mapping[target_node]
            server_config["addr"] = (
                f"http://{target_node_ip}:{server_config['port']}/mcp"
            )
            local_node = int(os.environ["SLURM_NODEID"])
            if local_node != target_node:
                logging.info(
                    f"Skipping server {server_config['script']} on node {local_node}, target node is {target_node}, target addr is {server_config['addr']}"
                )
                continue
            else:
                if server_config["env"] != "base":
                    try:
                        result = subprocess.run(
                            ["conda", "info", "--envs"],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        if server_config["env"] not in result.stdout:
                            print(
                                f"Warning: Conda environment '{server_config["env"]}' not found for {server_config["script"]}"
                            )
                    except:  # noqa
                        print(
                            f"Warning: Could not verify conda environment '{server_config["env"]}' for {server_config["script"]}"
                        )

                process = start_server(
                    script_name=server_config["script"],
                    port=server_config["port"],
                    conda_env=server_config["env"],
                    device=server_config["device"],
                    host=target_node_ip,
                )
                if process:
                    # Verify process actually started
                    time.sleep(0.5)
                    if process.poll() is not None:
                        logging.error(
                            f"❌ Process {server_config['script']} exited immediately with code {process.returncode}"
                        )
                        stdout, stderr = process.communicate(timeout=1)
                        if stdout:
                            logging.error(f"Error stdout: {stdout}")
                        if stderr:
                            logging.error(f"Error stderr: {stderr}")
                        continue

                    processes.append((process, server_config["script"]))
                    thread = threading.Thread(
                        target=read_output,
                        args=(process, server_config["script"], output_queue),
                        daemon=True,
                    )
                    thread.start()
                    output_threads.append(thread)
                    logging.info(
                        f"✅ Started process for {server_config['script']} (PID: {process.pid})"
                    )
                else:
                    logging.error(f"❌ Failed to start {server_config['script']}")

        # if not processes:
        #     raise RuntimeError("No servers were started successfully")

        # is_main_node = int(os.environ['SLURM_NODEID']) == 0

        # if is_main_node:
        logging.info("⏳ Waiting for servers to start...")
        logging.info("This is the main node, waiting for all servers to be ready...")
        servers_ready = await self.wait_for_servers()

        if not servers_ready:
            logging.error("Not all servers started successfully")
            raise RuntimeError("Server startup failed")

        def check_process_status(processes, output_queue):
            """check the status of all processes"""
            while True:
                time.sleep(2)  # check every 2 secs
                for process, script_name in processes:
                    if process.poll() is not None:  # process stopped
                        return_code = process.returncode
                        message = f"⚠️ Process exited with code {return_code}"
                        output_queue.put((script_name, "status", message))
                        logging.error(f"[{script_name}] {message}")

        status_thread = threading.Thread(
            target=check_process_status, args=(processes, output_queue), daemon=True
        )
        status_thread.start()

        logging.info("✅ Starting agent call...")
        t2 = time.time()
        logging.info(f"Time taken to start servers: {t2 - t1} seconds")

        return processes

    async def wait_for_servers(self, max_wait=3500, check_interval=10):
        server_configs = list(self.servers.values())
        """Wait for all servers to be ready by performing HTTP checks."""
        start_time = time.time()
        servers_status = {config["addr"]: False for config in server_configs}
        logging.info(f"Waiting for {len(server_configs)} servers to start...")

        while time.time() - start_time < max_wait:
            # Use asyncio.gather to check all servers concurrently in each interval
            tasks = []
            for config in server_configs:
                addr = config["addr"]
                if not servers_status[addr]:
                    tasks.append(check_server_ready(config["addr"]))
                else:
                    # If already ready, add a task that returns True
                    async def already_ready():
                        return True

                    tasks.append(already_ready())

            results = await asyncio.gather(*tasks)

            all_ready = True
            for i, config in enumerate(server_configs):
                addr = config["addr"]
                if not servers_status[addr] and results[i]:
                    servers_status[addr] = True
                    logging.info(
                        f"✅ Server on addr {addr} ({config['script']}) is ready"
                    )

                if not servers_status[addr]:
                    all_ready = False

            if all_ready:
                elapsed = time.time() - start_time
                logging.info(f"🎉 All servers ready in {elapsed:.1f} seconds")
                return True

            logging.info(
                f"Servers not ready after {time.time() - start_time:.1f}s. Retrying in {check_interval}s... Missing:{[addr for addr, ready in servers_status.items() if not ready]}",
            )
            await asyncio.sleep(check_interval)

        # Timeout - report which servers are not ready
        not_ready = [
            f"{c['script']} (port {c['port']})"
            for c in server_configs
            if not servers_status[c["addr"]]
        ]
        logging.error(f"⚠️ Timeout waiting for servers: {', '.join(not_ready)}")
        return False


async def check_server_ready(addr, timeout=20):
    """Check if a server is ready by making an MCP-compliant HTTP request"""
    # The header FastMCP looks for to avoid the 406 error
    headers = {"Accept": "application/json, text/event-stream"}

    try:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    addr,
                    headers=headers,  # Add this
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    # FastMCP returns 200 for a valid handshake
                    if response.status in [200, 400]:
                        return True
                    # Log the status if it's still not working
                    logging.info(f"Server at {addr} returned status: {response.status}")
                    return False
            except aiohttp.ClientError:
                return False
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logging.info(f"Error checking {addr}: {e}")
        return False


def read_output(process, script_name, output_queue):
    """Read the output from the process - both stdout and stderr simultaneously"""
    try:
        while process.poll() is None:
            # Use select to read from both streams without blocking
            readable, _, _ = select.select(
                [process.stdout, process.stderr], [], [], 0.1
            )

            for stream in readable:
                line = stream.readline()
                if line:
                    stream_type = "stdout" if stream == process.stdout else "stderr"
                    output_queue.put((script_name, stream_type, line.strip()))

        # Read any remaining output after process ends
        for line in process.stdout:
            output_queue.put((script_name, "stdout", line.strip()))
        for line in process.stderr:
            output_queue.put((script_name, "stderr", line.strip()))

    except Exception as e:
        output_queue.put((script_name, "error", f"Output reading error: {e}"))


def monitor_output_queue(output_queue):
    """Monitor and log output from all servers"""
    while True:
        try:
            script_name, stream_type, message = output_queue.get(timeout=1)
            if stream_type == "stderr":
                logging.warning(f"[{script_name}] {message}")
            elif stream_type == "status":
                logging.error(f"[{script_name}] {message}")
            elif stream_type == "error":
                logging.error(f"[{script_name}] {message}")
            else:
                logging.info(f"[{script_name}] {message}")
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error monitoring output: {e}")


def start_server(script_name, port=None, conda_env=None, device=0, host="localhost"):
    """start a FastMCP server"""
    try:
        # set env settings
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(device)

        # Pass port to the server if provided
        if port:
            env["PORT"] = str(port)

        if conda_env:
            # use conda run command to start conda environment
            command = [
                "conda",
                "run",
                "-n",
                conda_env,
                "--no-capture-output",
                "python",
                script_name,
            ]
            # Add port as command line argument if needed
        else:
            command = [sys.executable, script_name]

        command.extend(["--port", str(port), "--host", str(host)])

        logging.info(f"Starting command: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,  # line buffered
            text=True,
            env=env,
        )

        # Give it a moment and check if it started
        time.sleep(0.5)
        if process.poll() is not None:
            # Process already exited
            stdout, stderr = process.communicate(timeout=1)
            logging.error(
                f"Failed to start {script_name}: Process exited immediately with code {process.returncode}"
            )
            if stdout:
                logging.error(f"stdout: {stdout}")
            if stderr:
                logging.error(f"stderr: {stderr}")
            return None

        logging.info(
            f"✅ Started {script_name} on port {port} in environment '{conda_env}' (PID: {process.pid})"
        )
        return process
    except Exception as e:
        logging.error(f"❌ Failed to start {script_name}: {e}")
        import traceback

        traceback.print_exc()
        return None
