import sys, site
import langchain_mcp_adapters, mcp

print("Python executable:", sys.executable)
print("ENABLE_USER_SITE:", site.ENABLE_USER_SITE)
print("langchain_mcp_adapters from:", langchain_mcp_adapters.__file__)
print("mcp from:", mcp.__file__)