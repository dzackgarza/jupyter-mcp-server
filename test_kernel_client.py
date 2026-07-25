import asyncio
from jupyter_kernel_client import KernelClient
from jupyter_mcp_server.utils import wait_for_kernel_idle

async def main():
    kernel = KernelClient(server_url="http://localhost:8888", token="MY_TOKEN", kernel_id="dummy")
    print("Created kernel client")
    
if __name__ == "__main__":
    asyncio.run(main())
